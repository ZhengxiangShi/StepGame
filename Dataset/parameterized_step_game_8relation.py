import os
import numpy as np
import random
import argparse
from tqdm import tqdm
from template import get_sentence

# action_name = ['one step down', 'one step up', 'one step left', 'one step right']
# ac = {0: 1, 1: 0, 2: 3, 3: 2}
action_candidate = [[0,-1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, 1], [-1, 1], [1, -1]]

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def add_disconnected_noise(entity_name, length, story):
    diconnected_length = length
    disconnected_entity = entity_name
    for i in range(diconnected_length):
        agent_1 = disconnected_entity[i]
        agent_2 = disconnected_entity[i+1]
        action = np.random.randint(0, 8)
        noise_sentence = get_sentence(agent_2, action, agent_1)
        story.append(noise_sentence)
    return story

def add_irrelevant_noise(entity_name, length, d, story):
    irrelevant_length = length
    irrelevant_entity = entity_name
    for i in range(irrelevant_length):
        agent_1 = random.choice(list(d))
        agent_2 = irrelevant_entity.pop()
        d[agent_2] = 'pesudo_coordinate'
        action = np.random.randint(0, 8)
        noise_sentence = get_sentence(agent_2, action, agent_1)
        story.append(noise_sentence)
    return story

def add_supporting_noise(cur_entity, d, a, nhop, story):
    entity = cur_entity
    total_edge = 0
    att = 100

    # Choose two nodes whose distance is at least 2 
    while total_edge < 2 or total_edge > 7:
        random.shuffle(entity)
        noise_1 = entity[0]
        noise_2 = entity[1]
        x_diff = int(d[noise_2][0] - d[noise_1][0])
        y_diff = int(d[noise_2][1] - d[noise_1][1])
        total_edge = np.abs(x_diff) + np.abs(y_diff)
        att -= 1
        if att == 0: return story, 0

    remaining_entity = a[nhop+1:nhop+total_edge].copy() # The number of nodes used is (total_edge-1)

    cur_node = noise_2
    next_node = remaining_entity[0] 

    for edge in range(total_edge):
        if x_diff == 0 and y_diff == 0: break
        if x_diff != 0 and y_diff != 0:
            if np.random.randint(0, 2) == 1:
                if x_diff > 0:
                    x_diff -= 1
                    action = 2
                    sentence = get_sentence(next_node, action, cur_node)
                else:
                    x_diff += 1
                    action = 3
                    sentence = get_sentence(next_node, action, cur_node)
            else:
                if y_diff > 0:
                    y_diff -= 1
                    action = 0
                    sentence = get_sentence(next_node, action, cur_node)
                else:
                    y_diff += 1
                    action = 1
                    sentence = get_sentence(next_node, action, cur_node)
        elif x_diff == 0 and y_diff != 0:
            if y_diff > 0:
                y_diff -= 1
                action = 0
                sentence = get_sentence(next_node, action, cur_node)
            else:
                y_diff += 1
                action = 1
                sentence = get_sentence(next_node, action, cur_node)
        elif y_diff == 0 and x_diff != 0:
            if x_diff > 0:
                x_diff -= 1
                action = 2
                sentence = get_sentence(next_node, action, cur_node)
            else:
                x_diff += 1
                action = 3
                sentence = get_sentence(next_node, action, cur_node)
        else:
            print('Already overlapped.')
            assert next_node == noise_1

        story.append(sentence)

        if edge < (total_edge-1): # if edge = total_edge-1, no change is needed
            if edge < total_edge-2:
                cur_node = next_node
                next_node = remaining_entity[edge+1]
            elif edge == (total_edge-2):
                cur_node = next_node
                next_node = noise_1
            else:
                print('error')
        
    return story, total_edge

def generate_one_story(nhop, noise=True):
    """
    nhop: the maximum value of the potential number of reasoning steps.
    noise: whether add noise into samples
    """

    story = []
    d = {}
    current_position = [0, 0]
    candidates = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    ### Generate a story and its answer based on recored coordinates ###
    random.shuffle(candidates)
    entity = candidates[:nhop+1]
    d[entity[0]] = current_position.copy()
    for i in range(nhop):
        agent_1 = entity[i]
        agent_2 = entity[i+1]
        action = np.random.randint(0, 8)
        current_position[0] += action_candidate[action][0]
        current_position[1] += action_candidate[action][1]
        d[agent_2] = current_position.copy()

        sentence = get_sentence(agent_2, action, agent_1)
        story.append(sentence)

    random.shuffle(entity)
    ask_1 = entity.pop()
    ask_2 = entity.pop()
    difference = [0, 0]
    difference[0] = d[ask_2][0] - d[ask_1][0]
    difference[1] = d[ask_2][1] - d[ask_1][1]
    # Divide into 8 regions
    if difference[1] > 0:
        if difference[0] < 0:
            answer = 'upper-left'
        elif difference[0] > 0:
            answer = 'upper-right'
        else:
            answer = 'above'
    elif  difference[1] == 0:
        if difference[0] < 0:
            answer = 'left'
        elif difference[0] == 0:
            answer = 'overlap'
        else:
            answer = 'right'
    else:
        if difference[0] < 0:
            answer = 'lower-left'
        elif difference[0] > 0:
            answer = 'lower-right'
        else:
            answer = 'below'   
    q = 'What is the relation of the agent {} to the agent {}?'.format(ask_2, ask_1)
    

    ### Add distracting noise into a story ###
    if noise: 
        # Determine the number of nodes used for each type of noise
        supporting_nodes = 0
        num_edge_used = 0
        if nhop>3: # Generate supporting noise when there are more than three nodes in the original story
            story, num_edge_used = add_supporting_noise(entity, d, candidates, nhop, story) 
            if num_edge_used != 0: supporting_nodes = num_edge_used - 1
        try:
            disconnected_nodes = random.randint(2, int((nhop+1)/3)+1)
        except:
            disconnected_nodes = 2
        irrelevant_nodes = random.randint(1, int((nhop+1)/3)+1)
        assert (disconnected_nodes + irrelevant_nodes + supporting_nodes + (nhop + 1)) <= len(candidates)
        
        if num_edge_used != 0:
            disconnected_entity = candidates[nhop+num_edge_used:nhop+num_edge_used+disconnected_nodes+1]
            irrelevant_entity = candidates[nhop+num_edge_used+disconnected_nodes+1:nhop+num_edge_used+disconnected_nodes+1+irrelevant_nodes]
        else:
            disconnected_entity = candidates[nhop+1:nhop+2+disconnected_nodes]
            irrelevant_entity = candidates[nhop+2+disconnected_nodes:nhop+2+disconnected_nodes+irrelevant_nodes]        
        story = add_disconnected_noise(disconnected_entity, disconnected_nodes-1, story)
        story = add_irrelevant_noise(irrelevant_entity, irrelevant_nodes, d, story)
        return story, q, answer, (supporting_nodes, disconnected_nodes, irrelevant_nodes)

    return story, q, answer, (0, 0, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generator')
    # parser.add_argument('--nhop', type=int, default=10,
    #                     help='number of reasoning hops')
    parser.add_argument('--seed', type=int, default=111, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output_path', type=str, default='./data')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    # train_size_set = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
    train_size_set = [10000]*10
    test_size = 10000
    valid_size = 1000

    print('building datasets...')
    check_path(args.output_path)

    check_path(os.path.join(args.output_path,'clean'))
    check_path(os.path.join(args.output_path,'noise'))
    statistics = []
    for nhop in range(1, 11):
        print('Building dataset with {} hops'.format(nhop))
        
        print('building test datasets...')
        with open(os.path.join(args.output_path,'clean/qa{}_test.txt'.format(nhop)), 'w') as f_clean, open(os.path.join(args.output_path,'noise/qa{}_test.txt'.format(nhop)), 'w') as f_noise:
            s_noise = 0
            d_noise = 0
            i_noise = 0
            for _ in tqdm(range(test_size)):
                line = 1
                story, q, a, noise_node_num = generate_one_story(nhop)
                s_noise += noise_node_num[0]
                d_noise += noise_node_num[1]
                i_noise += noise_node_num[2]
                
                clean_story = story[:nhop].copy()
                random.shuffle(clean_story)
                for i in range(len(clean_story)):
                    f_clean.write(str(line) + ' ' + clean_story[i])
                    f_clean.write('\n')
                    line += 1
                f_clean.write(str(line) + ' ' + q+'\t'+a+'\t'+str(1))
                f_clean.write('\n')
                
                line = 1
                random.shuffle(story)
                for i in range(len(story)):
                    f_noise.write(str(line) + ' ' + story[i])
                    f_noise.write('\n')
                    line += 1
                f_noise.write(str(line) + ' ' + q+'\t'+a+'\t'+str(1))
                f_noise.write('\n')
            statistics.append('Test nhop {}: the average number of noise nodes is ({}, {}, {})'.format(nhop, s_noise/test_size, d_noise/test_size, i_noise/test_size))
            statistics.append('Total average: {}'.format((s_noise+d_noise+i_noise)/test_size))

        print('building train datasets...')
        with open(os.path.join(args.output_path,'clean/qa{}_train.txt'.format(nhop)), 'w') as f_clean, open(os.path.join(args.output_path,'noise/qa{}_train.txt'.format(nhop)), 'w') as f_noise:
            train_size = train_size_set[nhop-1]
            s_noise = 0
            d_noise = 0
            i_noise = 0
            for _ in tqdm(range(train_size)):
                line = 1
                story, q, a, noise_node_num = generate_one_story(nhop)
                s_noise += noise_node_num[0]
                d_noise += noise_node_num[1]
                i_noise += noise_node_num[2]

                clean_story = story[:nhop].copy()
                random.shuffle(clean_story)
                for i in range(len(clean_story)):
                    f_clean.write(str(line) + ' ' + clean_story[i])
                    f_clean.write('\n')
                    line += 1
                f_clean.write(str(line) + ' ' + q+'\t'+a+'\t'+str(1))
                f_clean.write('\n')
                
                line = 1
                random.shuffle(story)
                for i in range(len(story)):
                    f_noise.write(str(line) + ' ' + story[i])
                    f_noise.write('\n')
                    line += 1
                f_noise.write(str(line) + ' ' + q+'\t'+a+'\t'+str(1))
                f_noise.write('\n')
            statistics.append('Train nhop {}: the average number of noise nodes is ({}, {}, {})'.format(nhop, s_noise/train_size, d_noise/train_size, i_noise/train_size))
            statistics.append('Total average: {}'.format((s_noise+d_noise+i_noise)/train_size))

        print('building valid datasets...')
        with open(os.path.join(args.output_path,'clean/qa{}_valid.txt'.format(nhop)), 'w') as f_clean, open(os.path.join(args.output_path,'noise/qa{}_valid.txt'.format(nhop)), 'w') as f_noise:
            s_noise = 0
            d_noise = 0
            i_noise = 0
            for _ in tqdm(range(valid_size)):
                line = 1
                story, q, a, noise_node_num = generate_one_story(nhop)
                s_noise += noise_node_num[0]
                d_noise += noise_node_num[1]
                i_noise += noise_node_num[2]

                clean_story = story[:nhop].copy()
                random.shuffle(clean_story)
                for i in range(len(clean_story)):
                    f_clean.write(str(line) + ' ' + clean_story[i])
                    f_clean.write('\n')
                    line += 1
                f_clean.write(str(line) + ' ' + q+'\t'+a+'\t'+str(1))
                f_clean.write('\n')
                
                line = 1
                random.shuffle(story)
                for i in range(len(story)):
                    f_noise.write(str(line) + ' ' + story[i])
                    f_noise.write('\n')
                    line += 1
                f_noise.write(str(line) + ' ' + q+'\t'+a+'\t'+str(1))
                f_noise.write('\n')
            statistics.append('Valid nhop {}: the average number of noise nodes is ({}, {}, {})'.format(nhop, s_noise/valid_size, d_noise/valid_size, i_noise/valid_size))
            statistics.append('Total average: {}\n'.format((s_noise+d_noise+i_noise)/valid_size))

    with open(os.path.join(args.output_path, 'statistic.txt'), 'w') as f:
        for line in statistics:
            f.write(line)
            f.write('\n')

    print('Finished.')
