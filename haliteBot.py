from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np
from scipy.optimize import linear_sum_assignment

move_count = 5
double_move_count = 13
dx = [0,1,0,-1,0,2,0,-2,0,1,1,-1,-1]
dy = [1,0,-1,0,0,0,2,0,-2,1,-1,1,-1]

size = 21
size2 = size / 2

# distance in a direction
def distance_dir(a,b,direction,bound):
    if direction == 1:
        tdx = b[0] - a[0]
        if tdx < 0:
            tdx += size
    elif direction == 3:
        tdx = a[0] - b[0]
        if tdx < 0:
            tdx += size
    else:
        tdx = abs(a[0] - b[0])
        if tdx > size - tdx:
            tdx = size - tdx
        
    if direction == 0:
        tdy = b[1] - a[1]
        if tdy < 0:
            tdy += size
    elif direction == 2:
        tdy = a[1] - b[1]
        if tdy < 0:
            tdy += size
    else:
        tdy = abs(a[1] - b[1])
        if tdy > size - tdy:
            tdy = size - tdy
    
    if (direction == 1 or direction == 3) and tdy >= tdx + bound:
        tdx += size
        
    if (direction == 0 or direction == 2) and tdx >= tdy + bound:
        tdy += size
    
    if tdx == tdy:
        return tdx + tdy + 0.1
    return tdx + tdy

def dist(a, b):
    tdx = abs(a[0] - b[0])
    tdy = abs(a[1] - b[1])
    return min(tdx, size - tdx) + min(tdy, size - tdy)

def possible_moves(a, b):
    res = {}
    tdx = b[0] - a[0]
    tdy = b[1] - a[1]
    
    if tdx < -size2:
        tdx += size
    if tdx > size2:
        tdx -= size
    if tdy < -size2:
        tdy += size
    if tdy > size2:
        tdy -= size
    
    if tdx > 0:
        if tdx >= tdy and tdx >= -tdy:
            res[1] = 1.0
        else:
            res[1] = 0.9
    if tdy > 0:
        if tdy >= tdx and tdy >= -tdx:
            res[0] = 1.0
        else:
            res[0] = 0.9
    if tdx < 0:
        if tdx <= tdy and tdx <= -tdy:
            res[3] = 1.0
        else:
            res[3] = 0.9
    if tdy < 0:
        if tdy <= tdx and tdy <= -tdx:
            res[2] = 1.0
        else:
            res[2] = 0.9
            
    if tdx == 0 and tdy != 0:
        res[1] = 0.1
        res[3] = 0.1
    
    if tdx != 0 and tdy == 0:
        res[0] = 0.1
        res[2] = 0.1
            
    if len(res) == 0:
        res[4] = 1.0
    
    return res

move_to_action = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None]
     
def count_new_position(position, move):
    return ((position[0] + dx[move] + size) % size,(position[1] + dy[move] + size) % size)

def position_to_linear(position):
    return position[0]*size + position[1]

def linear_to_position(linear):
    return (int(linear/size),linear%size)

class Hoard:
    def __init__(self, ship, hoard, score):
        self.target = ship
        self.hoard = hoard
        self.score = score

class My_Ship:
    def __init__(self, ship):
        self.ship = ship
        self.intentions = [0.0,0.0,0.0,0.0,0.0]
        self.run_away = False
        self.going_back = False
        self.target = None
        self.target_priority = 0

halite_per_turn = {}

class Setting:
    def __init__(self):
        self.cell_halite = 0.69
        self.cell_halite_bonus = 0.45
        self.radius_bonus = 2
        self.cell_halite_multiplier = 1.33
        self.clean_time = 320
        self.collect_multiplier = 5.9
        self.zero_multiplier = 2.2
        self.attack_distance = 10.0
        self.spawning = 349
        self.far_radius = 2
        self.surround_divide = 2.6
        self.shipyard_radius = 4
        self.per_shipyard = 5.9
        self.zero_surround = 10

def my_agent17(observation, configuration, setting):
    board = Board(observation, configuration)

    sum_halite = 0
    cell_halite_count = 0
    for key, cell in board.cells.items():
        sum_halite += cell.halite
        if cell.halite > 1:
            cell_halite_count += 1
    
    per_ship = sum_halite * 0.25 * (400 - 2 * size - observation['step'])/15.0 / (len(board.ships) + 1)
    per_cell = sum_halite / cell_halite_count
    
    if len(halite_per_turn) == 0:
        for distance in range(size):
            prev = 0
            for harvest_time in range(1,21):
                temp = (1 - 0.75**harvest_time) / (harvest_time + distance)
                if temp <= prev:
                    halite_per_turn[distance] = prev
                    break
                prev = temp
    
    # players to attack
    players_to_attack = []
    for number, player in board.players.items():
        if player != board.current_player and observation['step'] > 300 and player.halite > board.current_player.halite:
            possible_my = (board.current_player.halite 
                           + 500 * (len(board.current_player.ships) + len(board.current_player.shipyards)))
            possible_him = player.halite + 500 * (len(player.ships) + len(player.shipyards))
            
            if possible_my > possible_him:
                players_to_attack.append(player)

            
    danger = [ [ [ 10000 for y in range(size) ] for x in range(size) ] for x in range(len(board.players)) ]
    zero_danger = [ [ False for y in range(size) ] for x in range(size) ]
    
    for (key,ship) in board.ships.items():
        for move in range(move_count):
            new_position = count_new_position(ship.position,move)
            for player in board.players:
                if ship.player.id != player and danger[player][new_position[0]][new_position[1]] > ship.halite:
                    danger[player][new_position[0]][new_position[1]] = ship.halite
                    if ship.player not in players_to_attack and ship.halite == 0:
                        zero_danger[new_position[0]][new_position[1]] = True
        
        
    base_attack = [ [ False for y in range(size) ] for x in range(size) ]
    base_bonus = [ [ 0 for y in range(size) ] for x in range(size) ]
            
    for key, shipyard in board.shipyards.items():
        if shipyard.player is not board.current_player and shipyard.player.halite >= 500:
            for player in board.players:
                if shipyard.player.id != player:
                    danger[player][shipyard.position[0]][shipyard.position[1]] = 0
                    if shipyard.player not in players_to_attack:
                        zero_danger[shipyard.position[0]][shipyard.position[1]] = True
        
        # my shipyard save danger
        if shipyard.player is board.current_player:
            # near base
            for i in range(size):
                for j in range(size):
                    temp_dist = dist(shipyard.position,[i,j]);
                    if temp_dist < setting.radius_bonus:
                        base_bonus[i][j] += setting.radius_bonus - temp_dist
    
    
    my_shipyard_dist = [ [ 10000 for y in range(size) ] for x in range(size) ]
    enemy_shipyard_dist = [ [ 10000 for y in range(size) ] for x in range(size) ]
    my_dist = [ [ 10000 for y in range(size) ] for x in range(size) ]
    enemy_dist = [ [ 10000 for y in range(size) ] for x in range(size) ]
    strong = [ [ [ 10000 for y in range(size) ] for x in range(size) ] for z in range(size) ]
    for i in range(size):
        for j in range(size):
            for key, shipyard in board.shipyards.items():
                distance = dist((i,j),shipyard.position)
                if shipyard.player == board.current_player:
                    if my_shipyard_dist[i][j] > distance:
                        my_shipyard_dist[i][j] = distance
                else:
                    if enemy_shipyard_dist[i][j] > distance:
                        enemy_shipyard_dist[i][j] = distance
                    
            for key, ship in board.ships.items():
                distance = dist((i,j),ship.position)
                if ship.player is not board.current_player:
                    if strong[i][j][distance] > ship.halite:
                        strong[i][j][distance] = ship.halite
                    if enemy_dist[i][j] > distance:
                        enemy_dist[i][j] = distance
                else:
                    if my_dist[i][j] > distance:
                        my_dist[i][j] = distance
            current_min = strong[i][j][0]
            for k in range(size):
                if strong[i][j][k] > current_min:
                    strong[i][j][k] = current_min
                current_min = strong[i][j][k]
                
            if my_shipyard_dist[i][j] < 3:
                base_attack[i][j] = True
                
                
    my_ships = []
    for ship in board.current_player.ships:
        my_ship = My_Ship(ship)
        if board.cells[ship.position].halite > 0:
            my_ship.intentions[4] -= 1.0
        my_ships.append(my_ship)

    for ship in my_ships:
        for move in range(move_count):
            new_position = count_new_position(ship.ship.position,move)
            if danger[board.current_player.id][new_position[0]][new_position[1]] < 10000:
                if danger[board.current_player.id][new_position[0]][new_position[1]] > ship.ship.halite:
                    ship.intentions[move] += 0.5
                if danger[board.current_player.id][new_position[0]][new_position[1]] == ship.ship.halite:
                    if (base_attack[new_position[0]][new_position[1]] == False
                        and (zero_danger[new_position[0]][new_position[1]] == True or ship.ship.halite > 0)):
                        ship.intentions[move] -= 64.0
                if danger[board.current_player.id][new_position[0]][new_position[1]] < ship.ship.halite:
                    ship.intentions[move] -= 64.0
                    if move == 4:
                        ship.run_away = True
  
    for ship in my_ships:
        if ship.ship.halite > 250 and ship.ship.halite + board.current_player.halite >= 500:
            all_wrong = True
            for move in range(move_count):
                new_position = count_new_position(ship.ship.position,move)
                if danger[board.current_player.id][new_position[0]][new_position[1]] > ship.ship.halite:
                    all_wrong = False
            if all_wrong:
                ship.ship.next_action = ShipAction.CONVERT
                ship.intentions[4] += 100.0
     
    cells_to_attack = {}
    
    # destroy near me
    for key,ship in board.ships.items():
        if ship.player is not board.current_player:
            for move in range(double_move_count):
                new_position = count_new_position(ship.position,move)
                for key2, shipyard in board.shipyards.items():
                    if shipyard.player is board.current_player and shipyard.position == new_position:
                        cells_to_attack[ship.position] = 250
    
    # biggest base
    shipyard_halite = {}
    for key,shipyard in board.shipyards.items():
        shipyard_halite[shipyard] = 0
    for key,ship in board.ships.items():
        if ship.player != board.current_player and ship.halite > 0:
            nearest_shipyard = None
            shipyard_dist = 100
            for key2, shipyard in board.shipyards.items():
                if shipyard.player == ship.player:
                    temp_dist = dist(ship.position,shipyard.position)
                    if temp_dist < shipyard_dist:
                        shipyard_dist = temp_dist
                        nearest_shipyard = shipyard
            if nearest_shipyard != None:
                shipyard_halite[nearest_shipyard] += ship.halite            

    # surround base
    for key, shipyard in board.shipyards.items():
        if shipyard.player is not board.current_player:
            target_priority = shipyard_halite[shipyard] / 100.0
            if shipyard.player in players_to_attack:
                target_priority = shipyard_halite[shipyard] + 1000.0
            for move in range(move_count):
                new_position = count_new_position(shipyard.position,move)
                cells_to_attack[new_position] = target_priority

    # can be surrounded        
    can_surround = [ [ 0 for y in range(size) ] for x in range(size) ]
    zero_ship_count = sum(map(lambda x : x.halite == 0, board.ships.values()))

    if zero_ship_count > setting.zero_surround:
        for (key, cell) in board.cells.items():
            best_shipyard = 1000
            for key2,shipyard in board.shipyards.items():
                if shipyard.player == board.current_player:
                    shipyard_surround = 1
                    enemies = [ 0 for x in range(size + size) ]
                    shipyard_dist = dist(key,shipyard.position)
                    for key3,ship in board.ships.items():
                        if ship.player != board.current_player and dist(shipyard.position,ship.position) <= shipyard_dist + 1:
                            enemies[dist(key,ship.position)] += 1
                    nearest_my = size + size - 1
                    
                    if zero_ship_count < 2 * setting.zero_surround:
                        for my_ship in my_ships:
                            if my_ship.ship.halite == 0:
                                my_dist_int = dist(key,my_ship.ship.position)
                                enemies[my_dist_int] -= 1
                                if nearest_my > my_dist_int:
                                    nearest_my = my_dist_int
                        enemies[nearest_my] += 1

                    current_enemy = 0
                    for i in range(shipyard_dist):
                        current_enemy += enemies[i]
                        if current_enemy > shipyard_surround:
                            shipyard_surround = current_enemy

                    if shipyard_surround < best_shipyard:
                        best_shipyard = shipyard_surround
            
            bonus = 1
            can_surround[key[0]][key[1]] = best_shipyard - bonus


    # do not get surrounded
    for my_ship in my_ships:
        for move in range(move_count):
            new_position = count_new_position(my_ship.ship.position,move)
            if can_surround[new_position[0]][new_position[1]] > 0:
                if my_ship.ship.halite > 0:
                    my_ship.intentions[move] -= 8.0 * can_surround[new_position[0]][new_position[1]]
                if my_ship.ship.halite == 0 and move == 4 and board.cells[(new_position[0],new_position[1])].halite > 0:
                    my_ship.intentions[move] -= 40.0

    # save surrounded
    for my_ship in my_ships:
        if my_ship.ship.halite > 0 and can_surround[my_ship.ship.position[0]][my_ship.ship.position[1]] > 0:
            cells_to_attack[my_ship.ship.position] = 250
            my_ship.run_away = True
                
    # surround
    hoards = []
    if len(my_ships) > 4:
        for key,ship in board.ships.items():
            if ship.player is not board.current_player:
                target_matrix = []
                for direction in range(4):
                    shipyard_dist = 1000
                    for key2,shipyard in board.shipyards.items():
                        if shipyard.player is ship.player:
                            temp_dist = distance_dir(ship.position,shipyard.position,direction,1)
                            if temp_dist < shipyard_dist:
                                shipyard_dist = temp_dist

                    target_row = [ 0.0 for y in range(len(my_ships)) ]
                    dist_limit = shipyard_dist

                    is_possible = False
                    for key2,ship2 in board.ships.items():
                        if ship.player != ship2.player and ship.halite > ship2.halite:
                            temp_dist = distance_dir(ship.position,ship2.position,direction,1)
                            if temp_dist < dist_limit:
                                dist_limit = temp_dist + 1
                                if ship2.player != board.current_player and temp_dist >= 5:
                                    is_possible = False
                                else:
                                    is_possible = True

                    if is_possible == False:
                        break

                    for i in range(len(my_ships)):
                        my_ship = my_ships[i]
                        if my_ship.ship.halite < ship.halite and my_ship.run_away == False:
                            temp_dist = distance_dir(ship.position,my_ship.ship.position,direction,1)
                            if temp_dist < dist_limit:
                                target_row[i] = 500.0 - temp_dist
                    target_matrix.append(target_row)

                if len(target_matrix) < 4:
                    continue

                target_row = [ 0.0 for y in range(len(my_ships)) ]     
                for i in range(len(my_ships)):
                    my_ship = my_ships[i]
                    if my_ship.ship.halite < ship.halite and my_ship.run_away == False:
                        temp_dist = dist(ship.position,my_ship.ship.position)
                        target_row[i] = (500.0 - temp_dist) / 2.0
                target_matrix.append(target_row)

                row_ind, col_ind = linear_sum_assignment(-np.array(target_matrix))
                overall_score = 0
                for index in range(len(row_ind)):
                    overall_score += target_matrix[row_ind[index]][col_ind[index]]

                hoard = []
                for index in range(len(row_ind)):
                    if target_matrix[row_ind[index]][col_ind[index]] == 0:
                        hoard.append(None)
                    else:
                        hoard.append(my_ships[col_ind[index]])

                hoards.append(Hoard(ship,hoard,overall_score))

    hoards.sort(key=lambda x: x.score, reverse=True)
    for hoard in hoards:
        if sum(map(lambda x : x is not None, hoard.hoard)) > 2:
            for direction in range(4):
                attack_ship = hoard.hoard[direction]
                if attack_ship is not None and attack_ship.target is None:
                    move1 = count_new_position(hoard.target.position,direction)
                    move2 = count_new_position(move1,direction)
                    dist1 = dist(attack_ship.ship.position,move1)
                    dist15 = 5

                    if hoard.hoard[4] is not None and hoard.hoard[4].target is None:
                        dist15 = dist(hoard.hoard[4].ship.position,move1)

                    if dist1 > 1:
                        attack_ship.target = move2
                    elif  dist1 == 1:
                        attack_ship.target = move1
                    else:
                        if dist15 == 1:
                            attack_ship.target = hoard.target.position
                            hoard.hoard[4].target = move1
                        elif board.cells[move1].halite == 0 or (can_surround[move1[0]][move1[1]] == 0
                            and board.cells[move1].halite / 4 + attack_ship.ship.halite < hoard.target.halite):
                            attack_ship.target = move1
                        else:
                            attack_ship.target = move2

            if hoard.hoard[4] is not None and hoard.hoard[4].target is None:
                hoard.hoard[4].target = hoard.target.position

                        
    # old hunt
    for key,ship in board.ships.items():
        if ship.player is not board.current_player:
            endangered = True
            for move in range(move_count):
                new_position = count_new_position(ship.position,move)
                if danger[ship.player.id][new_position[0]][new_position[1]] > ship.halite:
                    endangered = False
                    break
            if endangered:
                for move in range(move_count):
                    new_position = count_new_position(ship.position,move)
                    for my_ship in my_ships:
                        if my_ship.ship.halite < ship.halite and my_ship.target is None:
                            for my_move in range(move_count):
                                my_new_position = count_new_position(my_ship.ship.position,my_move)
                                if my_new_position == new_position:
                                    my_ship.intentions[my_move] += 8.0


    want_to_spawn = per_ship > setting.spawning or len(board.current_player.ships) == 0    
    
    # convert
    my_shipyard_count = sum(map(lambda x : x.player == board.current_player, list(board.shipyards.values())))
    target_shipyard_count = len(my_ships) / setting.per_shipyard - 1

    
    if my_shipyard_count < target_shipyard_count:
        if board.current_player.halite < 1500:
            want_to_spawn = False
        to_convert = None
        to_convert_score = 0
        for (key, cell) in board.cells.items():
            if (my_shipyard_dist[key[0]][key[1]] > setting.shipyard_radius and
                (my_shipyard_dist[key[0]][key[1]] < enemy_shipyard_dist[key[0]][key[1]] 
                or my_shipyard_dist[key[0]][key[1]] < 2 * setting.shipyard_radius
                or my_shipyard_dist[key[0]][key[1]] == 10000)
               and danger[board.current_player.id][key[0]][key[1]] > 0):
                cell_score = 0
                for (key2, cell2) in board.cells.items():
                    distance = dist(key,key2)
                    if distance <= setting.shipyard_radius and distance > 0:
                        esd = enemy_shipyard_dist[key2[0]][key2[1]]
                        msd = my_shipyard_dist[key2[0]][key2[1]]
                        rad = setting.shipyard_radius
                        if esd > rad:
                            esd = rad + 1
                        if msd > rad:
                            msd = rad + 1
                        
                        to_add = cell2.halite * (1 + rad - distance) / (4 + 4 * rad - esd - msd - msd - distance)

                        cell_score += to_add
                if cell_score > to_convert_score:
                    to_convert_score = cell_score
                    to_convert = key

        ship_to_convert = None
        for ship in my_ships:
            if ship.ship.position[0] == to_convert[0] and ship.ship.position[1] == to_convert[1]:
                ship_to_convert = ship
        if ship_to_convert != None:
            if ship_to_convert.ship.halite + board.current_player.halite >= 1000:
                ship_to_convert.ship.next_action = ShipAction.CONVERT
                ship_to_convert.intentions[4] += 100.0
        else:
            for move in range(move_count):
                new_position = count_new_position(to_convert,move)
                if move != 4:
                    cells_to_attack[new_position] = 500
                else:
                    cells_to_attack[new_position] = 1000

    cells_to_attack = {k: v  for k, v in sorted(cells_to_attack.items(), key=lambda item: -item[1])} 
    
    spawning = []
    
    # defend base
    if len(my_ships) > 0:
        for shipyard in board.current_player.shipyards:
            shipyard_danger = size+size
            for key,ship in board.ships.items():
                if ship.player is not board.current_player and dist(shipyard.position,ship.position) < shipyard_danger:
                    shipyard_danger = dist(shipyard.position,ship.position)

            defender_needed = True
            defender = None
            defender_score = -1
            for ship in my_ships:
                temp = dist(shipyard.position,ship.ship.position)
                
                if shipyard_danger - temp > 2 or (ship.ship.halite == 0 and shipyard_danger - temp > 1):
                    defender_needed = False
                    break
                    
                if ship.ship.halite > 0:
                    temp += 1
                
                temp_score = ship.ship.halite
                if ship.run_away:
                    temp_score += 500
                
                if temp <= shipyard_danger and temp_score > defender_score:
                    defender = ship
                    defender_score = temp_score

            if (defender_needed == True and defender == None and shipyard_danger == 1 and 
                target_shipyard_count > my_shipyard_count - 1):
                spawning.append(shipyard)

            if defender_needed == True and defender != None:
                defender.run_away = True
                if defender.ship.position == shipyard.position:
                    defender.intentions = [0.0,0.0,0.0,0.0,0.1]
                else:
                    for move, value in possible_moves(defender.ship.position,shipyard.position).items():
                        defender.intentions[move] += 16.0 * value    

    if want_to_spawn:
        spawning.extend(list(board.current_player.shipyards))
    
    # old targets 
    target_matrix = []
    target_ships = []
    for ship in my_ships:
        if ship.run_away == False and ship.ship.halite < 500:

            target_row = [ 0.0 for y in range(size*size) ]
            for (key, cell) in board.cells.items():
                halite_bound = per_cell * (setting.cell_halite + base_bonus[key[0]][key[1]] * setting.cell_halite_bonus)      
                if ship.ship.halite == 0:
                    halite_bound *= setting.cell_halite_multiplier
                if halite_bound > 400:
                    halite_bound = 400
                if observation['step'] > setting.clean_time or cell.halite > halite_bound:

                    distance = dist(cell.position, ship.ship.position)

                    if strong[key[0]][key[1]][distance] < ship.ship.halite:
                        continue

                    possible_gain = halite_per_turn[distance] * cell.halite

                    if setting.collect_multiplier * possible_gain > ship.ship.halite:

                        if enemy_dist[key[0]][key[1]] > my_shipyard_dist[key[0]][key[1]] + setting.far_radius and cell.halite < 400:
                            possible_gain /= (enemy_dist[key[0]][key[1]] - my_shipyard_dist[key[0]][key[1]] + 1 - setting.far_radius)

                        if ship.ship.halite > 0:
                            possible_gain *= setting.zero_multiplier
                        if can_surround[key[0]][key[1]] > 0:
                            possible_gain /= setting.surround_divide
                            if distance == 0 or ship.ship.halite > 0:
                                continue

                        target_row[position_to_linear(key)] = possible_gain

            for key,to_get in cells_to_attack.items():
                distance = dist(key, ship.ship.position)
                if distance > 0 and ship.ship.halite == 0:
                    target_row[position_to_linear(key)] = (to_get / 4 + setting.attack_distance) / distance

            if ship.target is not None:
                target_row[position_to_linear(ship.target)] = 500

            target_matrix.append(target_row)
            target_ships.append(ship)


    if len(target_matrix) > 0:
        row_ind, col_ind = linear_sum_assignment(-np.array(target_matrix))
        for index in range(len(row_ind)):
            if target_matrix[row_ind[index]][col_ind[index]] > 0:
                for move, value in possible_moves(target_ships[row_ind[index]].ship.position,linear_to_position(col_ind[index])).items():
                    if move == 4:
                        target_ships[row_ind[index]].intentions[move] += 5.0    
                    else:
                        target_ships[row_ind[index]].intentions[move] += (1.0 + 3.0*value)    

    
    if len(board.current_player.shipyards)>0:
        for ship in my_ships:
            nearest_shipyard = None
            nearest_dist = size
            for shipyard in board.current_player.shipyards:
                temp = dist(shipyard.position,ship.ship.position)
                if temp < nearest_dist:
                    nearest_dist = temp
                    nearest_shipyard = shipyard
            if nearest_shipyard:
                for move, value in possible_moves(ship.ship.position,nearest_shipyard.position).items():
                    if ship.ship.halite > 0:
                        ship.intentions[move] += 1.0 * value
                        if nearest_dist > 0 and observation['step'] > 400 - nearest_dist - 3 and value == 1.0:
                            ship.ship.next_action = move_to_action[move]
                            ship.going_back = True
                        
    
    target_matrix = []
    for ship in my_ships:
        target_row = [ 10000.0 for y in range(size*size) ]
        for move in range(move_count):
            target_row[position_to_linear(count_new_position(ship.ship.position,move))] = 1000.0 - ship.intentions[move]
        target_matrix.append(target_row)
    used = [ [ False for y in range(size) ] for x in range(size) ]
    if len(target_matrix) > 0:
        row_ind, col_ind = linear_sum_assignment(np.array(target_matrix))
        
        for index in range(len(row_ind)):
            new_position = linear_to_position(col_ind[index])
            move = list(possible_moves(my_ships[row_ind[index]].ship.position,new_position).keys())[0]
            if move < 4 and not my_ships[row_ind[index]].going_back:
                my_ships[row_ind[index]].ship.next_action = move_to_action[move]
            used[new_position[0]][new_position[1]] = True
            
    if len(board.current_player.shipyards)==0:
        my_halite = 0
        for ship in my_ships:
            my_halite += ship.ship.halite
        if my_halite > 500 or want_to_spawn:
            best_ship = None
            best_loss = 0
            for ship in board.current_player.ships:
                if ship.halite + board.current_player.halite > 500:
                    loss = 0
                    for ship2 in board.current_player.ships:
                        loss += dist(ship.position,ship2.position) * ship2.halite
                    if best_ship == None or loss < best_loss:
                        best_loss = loss
                        best_ship = ship
            if best_ship:
                best_ship.next_action = ShipAction.CONVERT   

    
    if len(my_ships) < 50:
        available_halite = board.current_player.halite
        for harbor in spawning:
            if not used[harbor.position.x][harbor.position.y] and available_halite >= 500:
                available_halite -= 500
                harbor.next_action = ShipyardAction.SPAWN
                used[harbor.position.x][harbor.position.y] = True
   
    return board.current_player.next_actions

def my_agent(observation,configuration):
    return my_agent17(observation, configuration,Setting())
