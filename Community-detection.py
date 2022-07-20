import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from networkx.algorithms import community as comm
from networkx.algorithms import centrality as cntr
import os

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

########################################################################################
############################## MY ROUTINES LIBRARY STARTS ##############################
########################################################################################

# SIMPLE ROUTINE TO CLEAR SCREEN BEFORE SHOWING A MENU...
def my_clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

# CREATE A LIST OF RANDOMLY CHOSEN COLORS...
def my_random_color_list_generator(REQUIRED_NUM_COLORS):

    my_color_list = [   'red',
                        'green',
                        'cyan',
                        'brown',
                        'olive',
                        'orange',
                        'darkblue',
                        'purple',
                        'yellow',
                        'hotpink',
                        'teal',
                        'gold']

    my_used_colors_dict = { c:0 for c in my_color_list }     # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])

    else: # REQUIRED_NUM_COLORS > len(my_color_list)
        constructed_color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(REQUIRED_NUM_COLORS)]

    return(constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G,fb_nodes_colors,fb_links_colors,fb_links_styles,graph_layout,node_positions):
    plt.figure(figsize=(10,10))

    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:   #DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G,
        with_labels=True,           # indicator variable for showing the nodes' ID-labels
        style=fb_links_styles,      # edge-list of link styles, or a single default style for all edges
        edge_color=fb_links_colors, # edge-list of link colors, or a single default color for all edges
        pos = node_positions,       # node-indexed dictionary, with position-values of the nodes in the plane
        node_color=fb_nodes_colors, # either a node-list of colors, or a single default color for all nodes
        node_size = 100,            # node-circle radius
        alpha = 0.9,                # fill-transparency
        width = 0.5                 # edge-width
        )
    plt.show()

    return(node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ###
########################################################################################
def my_menu_graph_construction(G,node_names_list,node_positions):

    my_clear_screen()

    breakWhileLoop  = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN
        + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)

        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102    # this is the maximum number of links in the fb-food-graph data set...

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else: #...len(my_option_list) == 2...
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                else:
                    # LOAD GRAPH FROM DATA SET...
                    G,node_names_list =_4096_4272_read_graph_from_csv(NUM_LINKS)
                    print(  "\tConstructing the FB-FOOD graph with n =",G.number_of_nodes(),
                            "vertices and m =",G.number_of_edges(),"edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else: # ...len(my_option_list) <= 3...
                if len(my_option_list) == 1:
                    NUM_NODES = 100                     # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES # DEFAULT VALUE FOR ER_EDGE_PROBABILITY...

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1,NUM_NODES) # AVOID DIVISION WITH ZERO...

                else: # ...NUM_NODES == 3...
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC +    "\tConstructing random Erdos-Renyi graph with n =",G.number_of_nodes(),
                                            "vertices and edge probability p =",ER_EDGE_PROBABILITY,
                                            "which resulted in m =",G.number_of_edges(),"edges.")

                    node_names_list = [ x for x in range(NUM_NODES) ]

        elif my_option_list[0] == 'P':                  # PLOT G...
            print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'     # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2:
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else: # ...len(my_option_list) == 3...
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y','y','N','n']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y','Y']:
                        node_positions = []         # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True

        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return(G,node_names_list,node_positions)

########################################################################################
# MENU 2: detect communities in the constructed graph
########################################################################################
def my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples):

    breakWhileLoop = False

    while not breakWhileLoop:
            print(bcolors.OKGREEN
                + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

            my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

            if my_option_list[0] == 'RE':                    # 2.1: ADD RANDOM EDGES TO NODES...

                if len(my_option_list) > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 1:
                        NUM_RANDOM_EDGES = 1                # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    elif len(my_option_list) == 2:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    else:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = float(my_option_list[2])

                    # CHECK APPROPIATENESS OF INPUT AND RUN THE ROUTINE...
                    if NUM_RANDOM_EDGES-1 not in range(5):
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        #G =
                        _4096_4272_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)

            elif my_option_list[0] == 'H':                  #2.2: ADD HAMILTON CYCLE...

                    #G =
                    _4096_4272_add_hamilton_cycle_to_graph(G,node_names_list)

            elif my_option_list[0] == 'P':                  # 2.3: PLOT G...
                print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )

                if len(my_option_list) > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) <= 1:
                        graph_layout = 'spring'     # ...DEFAULT graph_layout value...

                    else: # ...len(my_option_list) == 2...
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        if len(my_option_list) == 2:
                            node_positions = []         # ...ERASE previous node positions...

                        node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

            elif my_option_list[0] == 'C':      # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        alg_choice  = 'N'            # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        alg_choice  = str(my_option_list[1])
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    else: # ...NUM_OPTIONS == 3...
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])

                    # CHECKING CORRECTNESS OF GIVWEN PARAMETERS...
                    if alg_choice == 'N' and graph_layout in ['spring','circular','random','shell']:
                        _4096_4272_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)
                        #print(nx.number_connected_components(G),"connected components AFTER DEF.")
                        #print("len(community_tuples) is ",len(community_tuples)  )
                    elif alg_choice == 'O'and graph_layout in ['spring','circular','random','shell']:
                        _4096_4272_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)

                    else:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            elif my_option_list[0] == 'D':          # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
                NUM_OPTIONS = len(my_option_list)
                NUM_NODES = G.number_of_nodes()
                NUM_COMPONENTS = nx.number_connected_components(G)
                MAX_NUM_DIVISIONS = min( 8*NUM_COMPONENTS , np.floor(NUM_NODES/4) )

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        number_of_divisions = 2*NUM_COMPONENTS      # DEFAULT number of communities to look for
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring

                    else: #...NUM_OPTIONS == 3...
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])

                    # CHECKING SYNTAX OF GIVEN PARAMETERS...
                    if number_of_divisions < NUM_COMPONENTS or number_of_divisions > MAX_NUM_DIVISIONS:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already",NUM_COMPONENTS,"connected components." + bcolors.ENDC)
                        print(bcolors.WARNING + "\tProvide a number of divisions in { ",NUM_COMPONENTS,",",MAX_NUM_DIVISIONS,"}. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:

                        _4096_4272_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)



            elif my_option_list[0] == 'M':      # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
                _4096_4272_determine_opt_community_structure(G,hierarchy_of_community_tuples)


            elif my_option_list[0] == 'V':      # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:

                    if NUM_OPTIONS == 1:
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring

                    else: # ...NUM_OPTIONS == 2...
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        _4096_4272_visualize_communities(G,community_tuples,graph_layout,node_positions)

            elif my_option_list[0] == 'E':
                #EXIT the program execution...
                quit()

            else:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
    ### MENU 2 ENDS: detect communities in the constructed graph ###

########################################################################################
############################### MY ROUTINES LIBRARY ENDS ###############################
########################################################################################

########################################################################################
########################## _4096_4272 ROUTINES LIBRARY STARTS ##########################
# FILL IN THE REQUIRED ROUTINES FROM THAT POINT ON...
########################################################################################

########################################################################################

#This function reads the .edges file that containes the edges between the nodes,
#keeps the first NUM_LINKS edges and filters out the loops, edges with the same start and ending
#and creates he node_names_list that contains the nodes of the graph G.
#and the graph G
def _4096_4272_read_graph_from_csv(NUM_LINKS):


    #create the fb_nodes_dataframe
    fb_links_df = pd.read_csv('fb-pages-food.edges') #reads the .edges file
    fb_links_df=fb_links_df.head(NUM_LINKS)  #returns the first NUM_LINKS rows of the .edges file

    # update the fb_nodes_dataframe without loops
    fb_links_loopless_df =fb_links_df[fb_links_df['node_1'] != fb_links_df['node_2']]

    #create the node_names_list by combining the "node_1" and "node_2" arrays of tha file, so that the list has all teh nodes of the graph
    nodes_array = np.concatenate((fb_links_df['node_1'],fb_links_df['node_2']))
    node_names_list = (list(set(nodes_array)))#contains the nodes of the graph to be created


    #create graph from dataframe without loops
    G= nx.from_pandas_edgelist(fb_links_loopless_df, 'node_1', 'node_2',create_using= nx.Graph())

    return (G,node_names_list)

######################################################################################################################
# ...(a) _4096_4272 IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
#this function implements the girvan_newman algorithm to divide the biggest community
#of graph G, into to subgraphs and fills the community_tuples list with tuples, each
#containing the nodes of each community of the graph G.
def _4096_4272_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions):

    start_time = time.time()

    #community_tuples to be filled with tuples , each representig a community in the graph after FIRST substitution of a biggest community by two smaller.
    community_list=[]
    community_list=(sorted(nx.connected_components(G),key=len,reverse=True)) #list containg the communitites sorted , from biggest to smallest
    LC=community_list[0] #biggest community

    community_list.pop(0) #delete the biggest community from list
    GCC = G.subgraph(LC).copy()  #create graph o fbiggest subcommunity
    while nx.is_connected(GCC): #delete edges as long biggest subcommunitie is not connected
        betweenness= nx.edge_betweenness_centrality(GCC) #find betweenness of edges
        max_value = max(betweenness,key=betweenness.get) #find edge with biggest betweenness
        GCC.remove_edge(max_value[0], max_value[1]) #remove edge with biggest betweeness fro unfrozen_GCC
        G.remove_edge(max_value[0], max_value[1]) #and G

    G_community_list_of_lists = list(sorted(nx.connected_components(G))) #list containing al communities of graph G AFTER division of biggest community
    for com in G_community_list_of_lists:
        community_tuples.append(tuple(com)) #create community_tuples list containing a tuple for each community of G

    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")

######################################################################################################################
# ...(b) USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
#This function implements the girvan_newman algorithm using the girvan_newman() function
#fills the community_tuples list with tuples, each containing the nodes of each community
#of the graph G
#BUG: THIS FUNCTION WILL NOT DIVIDE THE BIGGER COMMUNITY INTO A SMALLER ONES
#IT WILL JUST CREATE THE NODE LIST WITH THE RIGHT COMMUNITIES
def _4096_4272_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions):

    start_time = time.time()

    #create communities of the graph G using the girvan_newman() fuction from the networkx library

    comp=comm.girvan_newman(G) #create the generator comp
    community_tuples.append(tuple(sorted(c) for c in next(comp))) #find the communities of the graph generated by the comp generator and add them as TUPLES in community_tuples


    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")

######################################################################################################################

#this function uses the community_tuples that was created from _4096_4272_one_shot_girvan_newman_for_communities()
#so it needs to run after it , so that no problems happen.
#This function diviedes graph G into number_of_divisions communities and creates triples.
#Each triple contains the graph tha was just divided and the noed list of thesubgraphs that derived from the first
def _4096_4272_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions):


    start_time = time.time()

    community_list = sorted(community_tuples, key=len, reverse=True) #create a sorted copy of community_tuples, needed for this function
    hierarchy_of_community_tuples.append(tuple(community_list)) #add the TUPLE community_list in the first cell of hierarchy_of_community_tuples list
    a=-1
    while not(a==1 or a==0):
        a=int(input("\tPress 0 if you choose to compute betweenness centrality from all nodes. Otherwise press 1."))
    sub=0
    random_nodes=[]
    if int(a) == 1 :
        sub=int(input("\tProvide the percetage of subset of nodes to use."))
    #REPEAT UNTIL ENOUGH CLUSTERS HAVE BEEN CREATED
    while int(nx.number_connected_components(G))<int(number_of_divisions):
        community_list = sorted(nx.connected_components(G), key=len, reverse=True) #update community_tuple list everytime a new cluster is created
        LC=community_list[0] #LC keeps the nodes of the biggest cluster of graph G
        community_list.pop(0) #remove the biggets cluster from community_list, becouse it is going to be divided
        #if graph G is TOO SMALL to create number_of_divisions clusters to begin with , do not try to divide cluster with one node
        if len(LC)==1:
            break
        GCC = G.subgraph(LC).copy() #GCC is the biggest subgraph

        #if user chose to give a percetage of root nodes to compute betweenness centrality of edges
        if int(a) == 1 :
            num=len(GCC.nodes())*sub//100 #find number of nodes of those that belong to gcc, to use as root nodes
            root_nodes = random.sample(list(GCC.nodes()), num) # root_nodes has a sample of GCC nodes, bases on the num number

        else:
            root_nodes=list(GCC.nodes()) #in case the user did not give a percetage use al the GCC root nodes

        #repeat until GCC is not connected
        while nx.is_connected(GCC):
            betweenness=nx.edge_betweenness_centrality_subset(GCC,root_nodes,list(GCC.nodes())) # find betweeness of the edges
            max_value = max(betweenness,key=betweenness.get) #keep the edge with biggest betweeness
            GCC.remove_edge(max_value[0], max_value[1]) # remove the edge with biggest betweeness from GCC
            G.remove_edge(max_value[0], max_value[1]) # remove the edge with biggest betweeness from G


        components = nx.connected_components(GCC) #components contains the GCC componets after division
        triple=[G.subgraph(LC)] #the first item of the triple is the GCC graph that was just divided
        for component in components:
            triple.append(component) #triple ALSO contains the componets created from GCC

        hierarchy_of_community_tuples.append(tuple(triple)) #finaly add to the hierarchy_of_community_tuples the triple tha was just made.


    end_time = time.time()
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tComputation of HIERARCHICAL BIPARTITION of G in communities, "
                        + "using the BUILT-IN girvan-newman algorithm, for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")

######################################################################################################################
#this function will find to modality of each partition and
#return the biggest one
#BUG: THE GRAPH DISPLAY OF THE PARTITION WITH THE BIGGEST MODALITY IS NOT IMPLEMENTED###########
def _4096_4272_determine_opt_community_structure(G,hierarchy_of_community_tuples):

    max_modularity_value=nx.algorithms.community.modularity(G,hierarchy_of_community_tuples[0]) #keep the modality of of the first set o communities
    max_partition=hierarchy_of_community_tuples[0] #keep the first partition in the hierarchy_of_community_tuples
    modalities=[max_modularity_value] #array that will determin the y-axis of the graph
    #modalities=-1
    k=len(hierarchy_of_community_tuples[0])
    communities_num=[k] #array that will determin the x-axis of the graph
    #compute modlity of each partition and find the biggest one
    for current_partition in hierarchy_of_community_tuples[1:]:
         current_modularity_value=nx.algorithms.community.modularity(current_partition[0],current_partition[1:])
         modalities.append(current_modularity_value)
         k=k+1
         communities_num.append(k)
         if current_modularity_value > max_modularity_value:
             max_partition = current_partition
             max_modularity_value = current_modularity_value

    #create the plot for modalities
    fig = plt.figure(figsize = (10, 5))
    plt.bar(communities_num, modalities, color ='maroon',width = 0.4)

    plt.xlabel("NUMBER OF COMMUNITIES IN PARTITION")
    plt.ylabel("MODALITY VALUE OF PARTITION")
    plt.title("Bar-Chart of Modality Values of Partitions in Hierarchy")
    plt.show()

    return (max_partition,max_modularity_value)

######################################################################################################################
#this function will add a hammilton cycle to the graph G so that , it becomes fully connected, if not already.
def _4096_4272_add_hamilton_cycle_to_graph(G,node_names_list):

    if nx.is_connected(G)==False: #if graph not connected
        for node1 in node_names_list:
            for node2 in node_names_list:
                if node1 != node2 and (G.has_edge(node1,node2)== False): #if two different nodes are found that are not connected
                    G.add_edge(node1,node2) #add the new edge to the graph


######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH...
######################################################################################################################
#this function will add up to NUM_RANDOM_EDGES edges from every node to random non_neighbors of the node
def _4096_4272_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY):
    for node in node_names_list:
        non_neighbors=list(nx.non_neighbors(G, node)) #create a list of nodes which are NOT neigbors of the node
        if non_neighbors != []:
            for attempts in range(int(NUM_RANDOM_EDGES)): #for up to NUM_RANDOM_EDGES attempts
                if float(random.random()) > float(EDGE_ADDITION_PROBABILITY): #for EDGE_ADDITION_PROBABILITY propability
                    i=random.randint(0,len(non_neighbors)-1) #find a random edge from the non_neighbors list
                    G.add_edge(node,non_neighbors[i])#and add the edge



######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
#this function will assign the correct color to each node of each community_list
#and display the colorful graph
def _4096_4272_visualize_communities(G,community_tuples,graph_layout,node_positions):

    number_of_communities=nx.number_connected_components(G) #keep number of communities of graph G
    list_of_communities=nx.connected_components(G) #keep communities of graph G
    list_of_colors=my_random_color_list_generator(number_of_communities) #generate a random list of colors, one for each community
    i=0
    node_colors=[]
    #create node_colors list that store the correct color assigned to each node. The nodes of each
    #community will have the same color
    for community in list_of_communities:
        for node in community:
            node_colors.append(list_of_colors[i])
        i=i+1
    my_graph_plot_routine(G,node_colors,'blue','solid',graph_layout,node_positions) #visualize the communities with colors
########################################################################################
########################### _4096_4272 ROUTINES LIBRARY ENDS ###########################
########################################################################################


########################################################################################
############################# MAIN MENUS OF USER CHOICES ###############################
########################################################################################

############################### GLOBAL INITIALIZATIONS #################################
G = nx.Graph()                      # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
node_names_list = []
node_positions = []                 # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []               # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES

G,node_names_list,node_positions = my_menu_graph_construction(G,node_names_list,node_positions)

my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples)
