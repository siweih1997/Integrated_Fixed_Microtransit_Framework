# -------------------------------------------------------------------------------------------------------------------- #
# external imports
# ----------------
import sys
import traceback
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
import time
import datetime
import get_microtransit_skims as mt
import csv
import numpy as np
# src imports
# -----------
import src.misc.config as config
from src.misc.init_modules import load_simulation_environment
from src.misc.globals import *

import inte_sys_mode_choice
import convergence_test as conv_test
import network_algorithms as n_a
import get_auto_skims as auto
import get_walk_transit_skims as wt
import binary_choice as b_c
import random
import update_network_files as update
import output_performance_metrics as output_metrics
import folder_directory as fld_dir
import testing_scenario_creation as tst_scen_create
import pre_process.dictionary_initialization as dict_init
# main functions
# --------------
def run_single_simulation(scenario_parameters):
    SF = load_simulation_environment(scenario_parameters)
    if scenario_parameters.get("bugfix", False):
        try:
            SF.run()
        except:
            traceback.print_exc()
    else:
        SF.run()


def run_scenarios(constant_config_file, scenario_file, n_parallel_sim=1, n_cpu_per_sim=1, evaluate=1, log_level="info",
                  keep_old=False, continue_next_after_error=False):
    """
    This function combines constant study parameters and scenario parameters.
    Then it sets up a pool of workers and starts a simulation for each scenario.
    The required parameters are stated in the documentation.

    :param constant_config_file: this file contains all input parameters that remain constant for a study
    :type constant_config_file: str
    :param scenario_file: this file contain all input parameters that are varied for a study
    :type scenario_file: str
    :param n_parallel_sim: number of parallel simulation processes
    :type n_parallel_sim: int
    :param n_cpu_per_sim: number of cpus for a single simulation
    :type n_cpu_per_sim: int
    :param evaluate: 0: no automatic evaluation / != 0 automatic simulation after each simulation
    :type evaluate: int
    :param log_level: hierarchical output to the logging file. Possible inputs with hierarchy from low to high:
            - "verbose": lowest level -> logs everything; even code which could scale exponentially
            - "debug": standard debugging logger. code which scales exponentially should not be logged here
            - "info": basic information during simulations (default)
            - "warning": only logs warnings
    :type log_level: str
    :param keep_old: does not start new simulation if result files are already available in scenario output directory
    :type keep_old: bool
    :param continue_next_after_error: continue with next simulation if one the simulations threw an error (only SP)
    :type continue_next_after_error: bool
    """
    assert type(n_parallel_sim) == int, "n_parallel_sim must be of type int"
    # read constant and scenario config files
    constant_cfg = config.ConstantConfig(constant_config_file)
    scenario_cfgs = config.ScenarioConfig(scenario_file)

    # set constant parameters from function arguments
    # TODO # get study name and check if its a studyname
    const_abs = os.path.abspath(constant_config_file)
    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

    if study_name == "scenarios":
        print("ERROR! The path of the config files is not longer up to date!")
        print("See documentation/Data_Directory_Structure.md for the updated directory structure needed as input!")
        exit()
    if constant_cfg.get(G_STUDY_NAME) is not None and study_name != constant_cfg.get(G_STUDY_NAME):
        print("ERROR! {} from constant config is not consistent with study directory: {}".format(constant_cfg[G_STUDY_NAME], study_name))
        print("{} is now given directly by the folder name !".format(G_STUDY_NAME))
        exit()
    constant_cfg[G_STUDY_NAME] = study_name

    constant_cfg["n_cpu_per_sim"] = n_cpu_per_sim
    constant_cfg["evaluate"] = evaluate
    constant_cfg["log_level"] = log_level
    constant_cfg["keep_old"] = keep_old

    # combine constant and scenario parameters into verbose scenario parameters
    for i, scenario_cfg in enumerate(scenario_cfgs):
        scenario_cfgs[i] = constant_cfg + scenario_cfg

    # perform simulation(s)
    print(f"Simulation of {len(scenario_cfgs)} scenarios on {n_parallel_sim} processes with {n_cpu_per_sim} cpus per simulation ...")
    if n_parallel_sim == 1:
        for scenario_cfg in scenario_cfgs:
            if continue_next_after_error:
                try:
                    run_single_simulation(scenario_cfg)
                except:
                    traceback.print_exc()
            else:
                run_single_simulation(scenario_cfg)
    else:
        if n_cpu_per_sim == 1:
            mp_pool = mp.Pool(n_parallel_sim)
            mp_pool.map(run_single_simulation, scenario_cfgs)
        else:
            n_scenarios = len(scenario_cfgs)
            rest_scenarios = n_scenarios
            current_scenario = 0
            while rest_scenarios != 0:
                if rest_scenarios >= n_parallel_sim:
                    par_processes = [None for i in range(n_parallel_sim)]
                    for i in range(n_parallel_sim):
                        par_processes[i] = mp.Process(target=run_single_simulation, args=(scenario_cfgs[current_scenario],))
                        current_scenario += 1
                        par_processes[i].start()
                    for i in range(n_parallel_sim):
                        par_processes[i].join()
                        rest_scenarios -= 1
                else:
                    par_processes = [None for i in range(rest_scenarios)]
                    for i in range(rest_scenarios):
                        par_processes[i] = mp.Process(target=run_single_simulation, args=(scenario_cfgs[current_scenario],))
                        current_scenario += 1
                        par_processes[i].start()
                    for i in range(rest_scenarios):
                        par_processes[i].join()
                        rest_scenarios -= 1

# -------------------------------------------------------------------------------------------------------------------- #
# ----> you can replace the following part by your respective if __name__ == '__main__' part for run_private*.py <---- #
# -------------------------------------------------------------------------------------------------------------------- #

# global variables for testing
# ----------------------------
MAIN_DIR = os.path.dirname(__file__)
MOD_STR = "MoD_0"
MM_STR = "Assertion"
LOG_F = "standard_bugfix.log"


# testing results of examples
# ---------------------------
def read_outputs_for_comparison(constant_csv, scenario_csv):
    """This function reads some output parameters for a test of meaningful results of the test cases.

    :param constant_csv: constant parameter definition
    :param scenario_csv: scenario definition
    :return: list of standard_eval data frames
    :rtype: list[DataFrame]
    """
    constant_cfg = config.ConstantConfig(constant_csv)
    scenario_cfgs = config.ScenarioConfig(scenario_csv)
    const_abs = os.path.abspath(constant_csv)
    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))
    return_list = []
    for scenario_cfg in scenario_cfgs:
        complete_scenario_cfg = constant_cfg + scenario_cfg
        scenario_name = complete_scenario_cfg[G_SCENARIO_NAME]
        output_dir = os.path.join(MAIN_DIR, "studies", study_name, "results", scenario_name)
        standard_eval_f = os.path.join(output_dir, "standard_eval.csv")
        tmp_df = pd.read_csv(standard_eval_f, index_col=0)
        tmp_df.loc[G_SCENARIO_NAME, MOD_STR] = scenario_name
        return_list.append((tmp_df))
    return return_list


def check_assertions(list_eval_df, all_scenario_assertion_dict):
    """This function checks assertions of scenarios to give a quick impression if results are fitting.

    :param list_eval_df: list of evaluation data frames
    :param all_scenario_assertion_dict: dictionary of scenario id to assertion dictionaries
    :return: list of (scenario_name, mismatch_flag, tmp_df) tuples
    """
    list_result_tuples = []
    for sc_id, assertion_dict in all_scenario_assertion_dict.items():
        tmp_df = list_eval_df[sc_id]
        scenario_name = tmp_df.loc[G_SCENARIO_NAME, MOD_STR]
        print("-"*80)
        mismatch = False
        for k, v in assertion_dict.items():
            if tmp_df.loc[k, MOD_STR] != v:
                tmp_df.loc[k, MM_STR] = v
                mismatch = True
        if mismatch:
            prt_str = f"Scenario {scenario_name} has mismatch with assertions:/n{tmp_df}/n" + "-"*80 + "/n"
        else:
            prt_str = f"Scenario {scenario_name} results match assertions/n" + "-"*80 + "/n"
        print(prt_str)
        with open(LOG_F, "a") as fh:
            fh.write(prt_str)
        list_result_tuples.append((scenario_name, mismatch, tmp_df))
    return list_result_tuples

def str2bool(string_, default='raise'):
    """
    Convert a string to a bool.

    Parameters
    ----------
    string_ : str
    default : {'raise', False}
        Default behaviour if none of the "true" strings is detected.

    Returns
    -------
    boolean : bool

    Examples
    --------
    # >>> str2bool('True')
    # True
    # >>> str2bool('1')
    # True
    # >>> str2bool('0')
    False
    """
    true = ['true', 't', '1', 'y', 'yes', 'enabled', 'enable', 'on']
    false = ['false', 'f', '0', 'n', 'no', 'disabled', 'disable', 'off']
    if string_.lower() in true:
        return True
    elif string_.lower() in false or (not default):
        return False
    else:
        raise ValueError('The value \'{}\' cannot be mapped to boolean.'
                         .format(string_))

def create_config_files(sc,study_area,repositioning,debug_mode,fleet_size):

    with open(sc, 'w+', newline='') as csvfile_config:
        if repositioning == True:
            fieldnames_config = ["op_module", "scenario_name", "rq_file", "op_fleet_composition","op_max_VR_con","forecast_f","op_repo_method","op_repo_horizons","op_repo_timestep"]
        else:
            fieldnames_config = ["op_module", "scenario_name", "rq_file", "op_fleet_composition"]
        writer_config = csv.DictWriter(csvfile_config, fieldnames=fieldnames_config)
        writer_config.writeheader()
        if repositioning == True:
            op_module="PoolingIRSOnly"
            scenario_name="example_pool_repo_AM_sc_1"
            rq_file="%s_debug_%s_fleetpy_demand.csv" % (str(study_area),str(debug_mode))
            op_fleet_composition="default_vehtype:"+str(fleet_size)
            op_max_VR_con=int(3)
            forecast_f=" "
            op_repo_method="AlonsoMoraRepositioning"
            op_repo_horizons=int(60)
            op_repo_timestep=int(300)
            writer_config.writerow({"op_module": op_module, "scenario_name": scenario_name, "rq_file": rq_file,
                                    "op_fleet_composition": op_fleet_composition,"op_max_VR_con": op_max_VR_con, "forecast_f": forecast_f,
                                    "op_repo_method": op_repo_method, "op_repo_horizons": op_repo_horizons,"op_repo_timestep": op_repo_timestep})
        else:
            op_module = "PoolingIRSOnly"
            scenario_name = "example_pool_irsonly_sc_1"
            rq_file = "%s_debug_%s_fleetpy_demand.csv" % (str(study_area), str(debug_mode))
            op_fleet_composition = "default_vehtype:" + str(fleet_size)
            writer_config.writerow({"op_module": op_module, "scenario_name": scenario_name, "rq_file": rq_file,
                                    "op_fleet_composition": op_fleet_composition})
    csvfile_config.close()


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    mp.freeze_support()

    start_time = datetime.datetime.now()
    pre_timeatfinished=start_time
    print("program starts here......start time is:", start_time)
    # zonal_partition=True
    input_para_folder="D:/Siwei_Micro_Transit/Bayesian_Optimization/Input_parameter"
    input_para_dir=os.path.join(input_para_folder,"input_parameter.csv")
    TRPartA_test_scenario_list=[]
    debug_mode_list=[]
    test_area=[]
    read_fleet_size=[]
    mile_meter = 1609.34

    ##
    with open(input_para_dir) as f:

        csvreader = csv.DictReader(f)
        for data in csvreader:
            #         print("line:",i)
            #             print(data)
            TRPartA = str2bool(str(data["TRPartA"]))
            TRPartA_test_scenario_list.append(int(data["TRPartA_test_scenario_list"]))
            debug_mode_list.append(str2bool(str(data["debug_mode_list"])))
            transit_fare = float(data["transit_fare ($)"])
            microtransit_dist_based_rate  = float(data["microtransit_dist_based_rate ($/mile)"])/mile_meter
            microtransit_start_fare  = float(data["microtransit_start_fare ($)"])
            test_area.append( str(data["test_area"]))
            BayesianOptimization = str2bool(str(data["BayesianOptimization"]))
            fleet_size=int(data["Fleet_size"])
            PkFareFactor=float(data["PkFareFactor"])
            OffPkFareFactor=float(data["OffPkFareFactor"])
            Fixed2MicroFactor=float(data["Fixed2MicroFactor"])
            Micro2FixedFactor=float(data["Micro2FixedFactor"])
            #PkFareFactor,OffPkFareFactor,Fixed2MicroFactor,Micro2FixedFactor

            read_fleet_size.append(fleet_size)

    f.close()

    zonal_partition_list=[False]
    indiv_converg_test=True
    # debug_mode=True
    # debug_mode_list = [False]
    # transit_fare=2.5
    # microtransit_dist_based_rate = 0.00123 #microtransit fare: 0.00123 dollar per meter, 1.97 dollar per mile
    # microtransit_start_fare=1.0

    # test_area = ["downtown_sd","lemon_grove"]
    # test_area = ["downtown_sd"]
    dt_sd_full_trnst_ntwk=True
    # TRPartA=True
    # test_area = ["downtown_sd"]
    # test_area = ["lemon_grove"]
    # TRPartA_test_scenario_list = [0, 1,2,3, 4, 5, 6, 7, 8, 9]
    # TRPartA_test_scenario_list=[0,1,4,5,6,7,8,9]
    # TRPartA_test_scenario_list = [2,3]
    # test_scenario=7

    iteration_debug = False
    if iteration_debug ==True:
        maximum_iteration = 15
    else:
        maximum_iteration = 10
    ###############
    #
    aggregation = "whole_region" #["census_track","whole_region"]
    convergence_gap=0.01
    #####################
    #Gas cost
    #####
    F_gas_per_mile = 0.350 #The fuel cost for the CNG buses averaged $0.35 per mile
    M_gas_per_mile = 0.305 # gasoline cost: 0.00019 dollar per meter, 0.305 dollar per mile



    #################
    ##########
    # Different operating hours testing scenarios
    operating_period_1 = ["AM", "PM"] #10 hr
    operating_period_2 = ["AM", "MD", "PM"] #15hr
    operating_period_3 = ["AM", "MD", "PM", "EV"] #19hr
    # operating_periods_scenarios=[operating_period_1,operating_period_2,operating_period_3]

    #####################
    # output_metrics
    mode_change_threshold=0.05
    # Different virtual stops testing scenarios
    # virtual_stop_scenarios=[50,75,100]
    # virtual_stop_scenarios = [75,100] #just for testing

    ###############################
    # Different headway testing scenarios
    # headway_scenarios=[20,30,60]

    #######################################
    #######################################
    #Create performance metrics dictonaries
    ###***Cost****###############################
    ################################
    ####1108:Siwei
    ################################
    test_repositioning=False

    # repositioning = True
    microtransit_running_scenarios = ["micro"]
    # microtransit_running_scenarios = ["micro"]
    repositioning_list=[True]
    # microtransit_only = True
    # microtransit="micro"
    if test_repositioning==False:
        network_folder="D:/Siwei_Micro_Transit/Data/0719_input"
        zonal_partition=False
        # for zonal_partition in zonal_partition_list:
        for debug_mode in debug_mode_list:
            if debug_mode == True:
                operating_periods_scenarios = [operating_period_2]  # just for testing
                # virtual_stop_scenarios = [75,100]  # just for testing
                # headway_scenarios = [30,60]  # just for testing
                # fleet_size_scenarios = [2,3,4]
                virtual_stop_scenarios = [75]  # just for testing
                headway_scenarios = [15]  # just for testing
                # fleet_size_scenarios = [1]
                fleet_size_scenarios = read_fleet_size
            else:
                # operating_periods_scenarios = [operating_period_1,operating_period_2]  # just for testing [operating_period_1,operating_period_2]
                # virtual_stop_scenarios = [75,100] #[75,100]
                # headway_scenarios = [15,30] #[30,60] # Downtown SD is every 15 minutes
                # # headway_scenarios = [20,30, 60]
                # fleet_size_scenarios = [10,15,20] #[10,20]
                operating_periods_scenarios = [operating_period_2]  # just for testing [operating_period_1,operating_period_2]
                virtual_stop_scenarios = [75]  # [75,100]
                headway_scenarios = [15]  # [30,60] # Downtown SD is every 15 minutes
                # headway_scenarios = [20,30, 60]
                # [2,3,4,5,6]
                # fleet_size_scenarios = [4]  # [5,10,15,20,30]
                fleet_size_scenarios = read_fleet_size

            for study_area in test_area:
                #######
                #05/14:Comment for now
                ####################
                demand_folder,initial_network_folder,final_network_folder,fleetpy_demand_folder,output_folder=fld_dir.determine_dolders(study_area,dt_sd_full_trnst_ntwk,zonal_partition,TRPartA,BayesianOptimization)
                new_fleetpy_demand = os.path.join(fleetpy_demand_folder,"%s_debug_%s_fleetpy_demand.csv" % (str(study_area), str(debug_mode)))
                #############
                ## transit information
                ##########
                transit_line_dist_list = output_metrics.get_transit_line_dist(study_area,dt_sd_full_trnst_ntwk)
                #################
                #################
                ###0514: comment for now
                transit_line_duration=output_metrics.get_transit_line_duration(study_area,dt_sd_full_trnst_ntwk,transit_line_dist_list)

                print("study_area",study_area,"transit_line_duration (route_id,duration (h))",transit_line_duration)
                scenario_count = 0


                # new_fleetpy_demand = os.path.join(fleetpy_demand_folder, "fleetpy_demand.csv")
                if debug_mode==True:
                    if TRPartA==True:
                        evaluation_file = os.path.join(output_folder, "%s_debug_evaluation_zonal_partition_%s.csv" % (str(study_area), str(zonal_partition)))
                        demand_file = os.path.join(demand_folder,"%s_debug_trips_nodes_study_area_with_beta.csv" % str(study_area))
                    else:
                        evaluation_file=os.path.join(output_folder,"%s_debug_evaluation_zonal_partition_%s.csv" % (str(study_area),str(zonal_partition)))
                        demand_file=os.path.join(demand_folder,"%s_debug_trips_nodes_study_area_with_beta.csv" % str(study_area))
                else:
                    if TRPartA==True:
                        evaluation_file = os.path.join(output_folder, "%s_evaluation_zonal_partition_%s.csv" % (str(study_area), str(zonal_partition)))
                        demand_file = os.path.join(demand_folder,"%s_trips_nodes_study_area_with_beta.csv" % str(study_area))
                    else:
                        evaluation_file = os.path.join(output_folder, "%s_evaluation_%s.csv" % (str(study_area),str(zonal_partition)))
                        demand_file=os.path.join(demand_folder,"%s_trips_nodes_study_area_with_beta.csv" % str(study_area))



            ##############################################
            #1108: read the initial network information
            #############################################
                MicroTransitNodes=OrderedDict()
                if study_area == "downtown_sd":
                    if dt_sd_full_trnst_ntwk==True:
                        for i in range(732):
                            MicroTransitNodes[872 + i] = i  # downtown_sd SD: microtransit node 773 - corresponding to - walking node 0 (microtransit node 0 in fleetpy)
                    else:
                        for i in range(732):
                            MicroTransitNodes[773 + i] = i #downtown_sd SD: microtransit node 773 - corresponding to - walking node 0 (microtransit node 0 in fleetpy)

                else:
                    for i in range(1099):
                        MicroTransitNodes[1171 + i] = i #lemon_grove: microtransit node 1171 - corresponding to - walking node 0 (microtransit node 0 in fleetpy)
                ########
                # headway_scenarios = [20,30,60]
                microtransit_setup_scenarios = ["micro", "non_micro","micro_only"]
                # microtransit_scenarios = ["micro", "non_micro"]
                time_periods = ["AM", "MD", "PM", "EV"]

                auto_network_dir=os.path.join(initial_network_folder,"auto_edges.csv")
                auto_network = n_a.read_super_network(auto_network_dir)

                ##############################
                #### Create initial and final network scenarios
                ###################
                ######################################################

                initial_super_network_dir, initial_super_network, final_super_network_dir, final_super_network=tst_scen_create.scen_create(microtransit_setup_scenarios,virtual_stop_scenarios,operating_periods_scenarios,time_periods,fleet_size_scenarios,headway_scenarios,initial_network_folder,final_network_folder)
                #################################################
                # 1108: prepare for the request files - demand side
                ###################################################
                agent_list_ = n_a.read_request(demand_file)
                tt_num_agents=len(agent_list_)
                print("study_area:", study_area,"downtown_sd_full_transit_network:",dt_sd_full_trnst_ntwk,"total number of agents:",tt_num_agents,"\n")

                transit_link_list=output_metrics.get_transit_link_info(study_area,dt_sd_full_trnst_ntwk)
                transit_link_vmt,transit_link_pax,off_micro_transit_link_pax,off_micro_transit_link_vmt=output_metrics.vmt_and_link_dict_creation(time_periods,transit_link_list)


                with open(evaluation_file, 'w+', newline='') as csvfile:

                    fieldnames = ["test_scen","dscrptn","study_area","repositioning","microtrasnit",'hdwy (min)', 'vir_stop (%)', 'flt_sz', 'op_periods',"tt_agents", "car_users", "car_mode_share (%)",
                                  "trsit_mode_users (W_M_F)","transit_mode_share (%)","pure_M_users","M_mode_share (%)","M_trips","pure_F_users","F_mode_share (%)","F_trips","pure_walk_users",
                                  "W_mode_share (%)","walk_users","M_pls_F_users","M_pls_F_mode_share (%)","F_oper_cost ($)", "M_oper_cost ($)", "F_revenue ($)","M_revenue ($)","Total_T_revenue ($)","tot_sub ($)",
                                  "sub_per_F_trip ($)","sub_per_F_rider ($)","sub_per_M_trip ($)","sub_per_M_rider ($)","sub_per_T_trip ($)","sub_per_T_rider ($)","sub_per_M_pax_mile ($/mi)","sub_per_F_pax_mile ($/mi)","sub_per_T_pax_mile ($/mi)",
                                  "sub_per_M_VMT ($/mi)","sub_per_F_VMT ($/mi)","sub_per_T_VMT ($/mi)",
                                  "tt_auto_gas_cost ($)","auto_gas_cost_per_mile ($/mi)",
                                  "avg_M_fare", "avg_F_fare", "avg_T_fare", "avg_auto_gas_cost",
                                  "tt_o_pckt_cost ($)","tt_mob_lgsm_inc_with_micro", "tt_gen_cost",
                                  "tt_mode_switch", "M_avg_wait_time (s)", "F_avg_wait_time (s)","avg_walk_time (s)","tt_walk_time (h)",
                                  "car_VMT (mi)","M_VMT (mi)","M_PMT (mi)","M_PMT/M_VMT","F_VMT (mi)","F_PMT (mi)","F_PMT/F_VMT","tt_VMT (mi)",
                                  "tt_walk_dist (mi)", "wghted_acc_emp_5_min","wghted_acc_emp_10_min","wghted_acc_emp_15_min", "M_util_rate (%)","M_veh_occ","M_avg_speed (mph)","cnvrg (iter, sm_sq_per_diff)"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for repositioning in repositioning_list:

                        ##########################
                        # store the auto shortest path result
                        ########################
                        agent_auto_visited_temp, agent_auto_time_visited_temp, agent_auto_dist_visited_temp, agent_auto_fare_visited_temp, agent_auto_path, agent_auto_time_path, agent_auto_time_path_all_ = dict_init.auto_sp_dict_init()

                        ##########################
                        # store the transit shortest path result
                        ########################
                        agent_visited_temp_F, agent_time_visited_temp_F, agent_dist_visited_temp_F, agent_fare_visited_temp_F, agent_path_F, agent_time_path_F, agent_transit_time_path_all_F, agent_transit_dist_path_all_F = dict_init.F_transit_sp_dict_init()

                        for test_scenario in TRPartA_test_scenario_list:
                            print("Zonal Partition (4 zones):",zonal_partition,"; Debug mode:",debug_mode,"; aggregation:",aggregation,"; mode_change_threshold:",mode_change_threshold,"; convergence_gap:",convergence_gap,"; maximum_iteration:",maximum_iteration,  "; F_gas_per_mile ($/mi):",F_gas_per_mile,
                            "; M_gas_per_mile ($/mi):", M_gas_per_mile,"; repositioning",repositioning,"; TRPartA:",TRPartA,"; Testing Scenario:",test_scenario,"; Bayesian Optimization:",BayesianOptimization,"; transit_fare ($):", transit_fare,"; microtransit_start_fare ($):",microtransit_start_fare,"; microtransit_dist_based_rate ($/mi):",microtransit_dist_based_rate*mile_meter,
                                  "; Peak fare factor (0~1):",PkFareFactor,"; Off-peak fare factor (0~1):",OffPkFareFactor,"; Fixed to Micro discount factor (0~1):",Fixed2MicroFactor,"; Micro to Fixed discount factor (0~1):",Micro2FixedFactor)


                            ###Siwei: 07/04/2024: comment
                            # ##########################
                            # # store the auto shortest path result
                            # ########################
                            # agent_auto_visited_temp,agent_auto_time_visited_temp,agent_auto_dist_visited_temp,agent_auto_fare_visited_temp,agent_auto_path,agent_auto_time_path,agent_auto_time_path_all_=dict_init.auto_sp_dict_init()
                            #
                            # ##########################
                            # # store the transit shortest path result
                            # ########################
                            # agent_visited_temp_F,agent_time_visited_temp_F,agent_dist_visited_temp_F,agent_fare_visited_temp_F,agent_path_F,agent_time_path_F,agent_transit_time_path_all_F,agent_transit_dist_path_all_F=dict_init.F_transit_sp_dict_init()

                            num_headway_scen = 0
                            for headway in headway_scenarios: #put bus design parameter in the most outer loop
                                tot_weighted_5_min_F,tot_weighted_10_min_F,tot_weighted_15_min_F=output_metrics.all_scenario_fixed_acc_calculation(headway,study_area,debug_mode,dt_sd_full_trnst_ntwk,test_scenario,TRPartA,BayesianOptimization)
                                num_headway_scen+=1
                                num_virstop_scen = 0
                                for virstop in virtual_stop_scenarios: #virtual stops - not necessarily influence the cost, but will influence the mobility
                                    num_virstop_scen+=1
                                    num_fleet_size_scen = 0
                                    for fleet_size in fleet_size_scenarios: #fleetsize - influence the cost & mobility
                                        num_fleet_size_scen+=1
                                        scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_study", "scenarios")
                                        num_operating_periods_scen = 0

                                        if repositioning ==True:
                                            if debug_mode == True:
                                                sc = os.path.join(scs_path, "%s_debug_example_ir_heuristics_repositioning_%s_veh.csv" % (str(study_area), str(fleet_size)))
                                            else:
                                                sc = os.path.join(scs_path, "%s_example_ir_heuristics_repositioning_%s_veh.csv" % (str(study_area), str(fleet_size)))
                                        else:
                                            if debug_mode == True:
                                                sc = os.path.join(scs_path, "%s_debug_example_ir_only_%s_veh.csv" % (str(study_area), str(fleet_size)))
                                                # "%s_debug_example_ir_only_%s_veh.csv" % (str(study_area), str(fleet_size)))
                                            else:
                                                sc = os.path.join(scs_path, "%s_example_ir_only_%s_veh.csv" % (str(study_area), str(fleet_size)))
                                        isExist = os.path.exists(sc)
                                        if isExist==False:
                                            create_config_files(sc, study_area, repositioning, debug_mode, fleet_size)

                                        for operating_periods in operating_periods_scenarios: #fleetsize - influence the cost & mobility
                                            num_operating_periods_scen+=1
                                            if operating_periods==["AM", "MD", "PM", "EV"]:
                                                M_operating_hrs=19
                                            elif operating_periods==["AM", "MD", "PM"]:
                                                M_operating_hrs = 15
                                            else:
                                                M_operating_hrs = 10
                                            F_operating_hrs = 19 # fixed route transit 19 hrs
                                            # route_duration = 60  # 60minutes
                                            # print("Current Scenario:",scenario_count,"th senario","....headway",headway,"(mins),","scen",num_headway_scen,
                                            #       "; virtual_stop:",virstop,"(%),","scen",num_virstop_scen,"; fleet_size",fleet_size,"(veh),","scen",num_fleet_size_scen,"; operating_periods",operating_periods,"scen,",num_operating_periods_scen, ";")


                                            F_transit_trips = os.path.join(fleetpy_demand_folder, "debug_%s_F_transit_demand_%s_op_hr_%s_hw_%s_virstop_%s.csv" % (str(debug_mode),str(fleet_size),str(M_operating_hrs), str(headway), str(virstop)))

                                            for microtransit_run in microtransit_running_scenarios:
                                                if microtransit_run == "micro_only" and num_headway_scen>1:
                                                    continue
                                                else:
                                                    if len(sys.argv) > 1:
                                                        run_scenarios(*sys.argv)
                                                    else:
                                                        import time
                                                        # touch log file
                                                        with open(LOG_F, "w") as _:
                                                            pass



                                                        # Base Examples IRS only
                                                        # ----------------------
                                                        # a) Pooling in ImmediateOfferEnvironment
                                                        ################################################
                                                        #1108: prepare the network files - supply side
                                                        ###################################################
                                                        print("Current Scenario:", scenario_count, "th senario", "....headway",headway, "(mins),", "scen", num_headway_scen,
                                                              "; virtual_stop:", virstop, "(%),", "scen", num_virstop_scen,"; fleet_size:", fleet_size, "(veh),", "scen", num_fleet_size_scen,
                                                              "; operating_periods:", operating_periods, "scen,",num_operating_periods_scen,"microtransit_scen:",microtransit_run,";")

                                                        iteration=0
                                                        iter_agent_choice_prob=OrderedDict()
                                                        iter_agent_selected_mode=OrderedDict()
                                                        # req=1000
                                                        # veh=5
                                                        # vir_stop=0.1
                                                        converged=False
                                                        # iter_mode_share=OrderedDict()

                                                        # Metrics of turn-off microtransit
                                                        off_micro_T_trips = 0
                                                        off_micro_W_trips = 0
                                                        off_micro_F_trips = 0
                                                        off_micro_auto_trips = 0
                                                        off_micro_agent_tot_walk_time = OrderedDict()
                                                        off_micro_agent_F_tot_wait_time = OrderedDict()
                                                        off_micro_agent_op_cost = OrderedDict()
                                                        off_micro_VMT_F = 0
                                                        off_micro_VMT_W = 0
                                                        off_micro_VMT_auto = 0
                                                        off_micro_agent_gen_cost = OrderedDict()
                                                        off_micro_ag_tot_tt = OrderedDict()
                                                        off_micro_ag_mode = OrderedDict()
                                                        # off_micro_ag_tot_tt_auto = OrderedDict()
                                                        off_micro_ag_tot_dist = OrderedDict()
                                                        off_micro_agent_ran_num = OrderedDict()
                                                        off_micro_agent_choice_prob = OrderedDict()
                                                        off_micro_c_tt_travel_time = OrderedDict()
                                                        off_micro_f_tt_travel_time = OrderedDict()
                                                        # off_micro_ag_VMT_F = OrderedDict()
                                                        # off_micro_agent_auto_tot_travel_time = OrderedDict()
                                                        off_micro_ag_VMT_W = OrderedDict()
                                                        transit_tt_VMT = 0




                                                        num_converged=0

                                                        pre_iter_prob_T = OrderedDict()
                                                        pre_iter_mode = OrderedDict()
                                                        pre_iter_M_info = OrderedDict()
                                                        pre_iter_F_info = OrderedDict()

                                                        ag_VMT_F = OrderedDict()
                                                        ag_VMT_M = OrderedDict()
                                                        ag_VMT_W = OrderedDict()

                                                        ag_num_W_trips = OrderedDict()
                                                        ag_num_M_trips = OrderedDict()
                                                        ag_num_F_trips = OrderedDict()

                                                        #agent level attributes
                                                        travel_time_delta = OrderedDict()
                                                        gen_cost_delta = OrderedDict()
                                                        mob_logsum_delta = OrderedDict()
                                                        agent_beta_fare = OrderedDict()

                                                        agent_O = OrderedDict()
                                                        agent_D = OrderedDict()
                                                        agent_rq_time = OrderedDict()
                                                        agent_choice_prob = OrderedDict()
                                                        agent_switch_mode = OrderedDict()
                                                        agent_ran_num = OrderedDict()
                                                        agent_M_tot_wait_time = OrderedDict()
                                                        agent_F_tot_wait_time = OrderedDict()
                                                        agent_M_tot_travel_time = OrderedDict()
                                                        agent_F_tot_travel_time = OrderedDict()

                                                        agent_M_tot_travel_dist = OrderedDict()
                                                        agent_F_tot_travel_dist = OrderedDict()

                                                        agent_tot_walk_time = OrderedDict()
                                                        agent_travel_time = OrderedDict()
                                                        agent_travel_dist = OrderedDict()
                                                        agent_op_cost = OrderedDict()  # agent_out-of-pocket cost
                                                        agent_op_cost_Fix  = OrderedDict()
                                                        agent_op_cost_Micro = OrderedDict()

                                                        agent_gen_cost = OrderedDict()

                                                        iter_sum_sq_per_diff=[]

                                                        agent_income = OrderedDict()
                                                        agent_transit_15min_acc = OrderedDict()


                                                        while converged==False:
                                                            print("iteration:",iteration)
                                                            iter_agent_choice_prob[iteration]=OrderedDict()
                                                            # iter_agent_selected_mode[iteration] = OrderedDict()
                                                            log_level = "info"

                                                            if repositioning == True:
                                                                cc = os.path.join(scs_path, "%s_constant_config_ir_repo.csv"% str(study_area))
                                                            else:
                                                                cc = os.path.join(scs_path, "%s_constant_config_ir.csv" % str(study_area))

                                                                # cc = os.path.join(scs_path, "constant_config_ir.csv")

                                                            print("Demand Component (choice model).....")
                                                            ###########################################3
                                                            # Initial dictionaries to store performance metrics

                                                            num_car_trips = 0
                                                            num_transit_users = 0
                                                            num_M_trips = 0
                                                            num_F_trips = 0
                                                            num_W_trips = 0

                                                            pure_M_user = 0
                                                            pure_F_user = 0
                                                            M_F_user = 0

                                                            #calculate VMT when micro transit is turned on
                                                            VMT_M = 0
                                                            VMT_F = 0
                                                            VMT_auto = 0
                                                            VMT_W = 0




                                                            #################################################
                                                            #Demand component: calculate how many travelers will choose transit, how many will choose auto
                                                            # Write into FleetPy's demand input directly
                                                            #######################################################
                                                            # Read the updated super-network file
                                                            ###############

                                                            # for microtransit in microtransit_setup_scenarios:
                                                            if microtransit_run == "micro_only":
                                                                for time_period in time_periods:
                                                                    if iteration == 0:
                                                                        initial_super_network[microtransit_run][virstop][M_operating_hrs] = n_a.read_super_network(initial_super_network_dir[microtransit_run][virstop][M_operating_hrs])
                                                                    else:
                                                                        final_super_network[microtransit_run][virstop][M_operating_hrs][time_period][fleet_size]=n_a.read_super_network(final_super_network_dir[microtransit_run][virstop][M_operating_hrs][time_period][fleet_size])
                                                            elif microtransit_run=="micro":
                                                                for time_period in time_periods:
                                                                    if iteration == 0:
                                                                        #[microtransit][headway][virstop][M_operating_hrs][time_period]
                                                                        initial_super_network[microtransit_run][headway][virstop][M_operating_hrs] = n_a.read_super_network(initial_super_network_dir[microtransit_run][headway][virstop][M_operating_hrs])
                                                                    else:
                                                                        try:
                                                                            final_super_network[microtransit_run][headway][virstop][M_operating_hrs][time_period][fleet_size]=n_a.read_super_network(final_super_network_dir[microtransit_run][headway][virstop][M_operating_hrs][time_period][fleet_size])
                                                                        except:
                                                                            raise Exception("iteration",iteration,"converged",converged,"....headway",headway, "(mins),", "microtransit", microtransit_run,"; virtual_stop:", virstop, "(%),", "time_period", time_period,"; fleet_size:", fleet_size, "(veh)," ,"; operating_periods:", operating_periods,";")
                                                            # else:
                                                            initial_super_network["non_micro"][headway]=n_a.read_super_network(initial_super_network_dir["non_micro"][headway])

                                                            change_mode_ag_list=[]
                                                            print("Demand Component.....assigning travelers to different modes...")
                                                            with open(F_transit_trips, 'w+', newline='') as csvfile_F:
                                                                fieldnames_F = ["depart_time", "start", "end", "request_id"]
                                                                writer_F = csv.DictWriter(csvfile_F, fieldnames=fieldnames_F)
                                                                writer_F.writeheader()

                                                                with open(new_fleetpy_demand, 'w+', newline='') as csvfile_M:
                                                                    fieldnames_M = ["rq_time", "start", "end", "request_id"]
                                                                    writer_M = csv.DictWriter(csvfile_M, fieldnames=fieldnames_M)
                                                                    writer_M.writeheader()

                                                                    M_rq_id_list = []
                                                                    for agent in agent_list_:
                                                                        change_mode=True  #change_mode is True, unless the probability difference is smaller than the mode_change_threshold

                                                                        origin = agent.rq_O
                                                                        dest = agent.rq_D
                                                                        rq_time = agent.rq_time
                                                                        rq_id = agent.rq_id

                                                                        agent_O[rq_id] = origin
                                                                        agent_D[rq_id] = dest
                                                                        agent_rq_time[rq_id] = rq_time
                                                                        agent_income[rq_id]= agent.income
                                                                        agent_transit_15min_acc[rq_id] = agent.transit_15min_acc
                                                                        agent_beta_fare[rq_id] = agent.bt_t_fr

                                                                        if rq_time <= (10 * 3600):
                                                                            time_period = "AM"
                                                                        elif rq_time <= (15 * 3600):
                                                                            time_period = "MD"
                                                                        elif rq_time <= (20 * 3600):
                                                                            time_period = "PM"
                                                                        else:
                                                                            time_period = "EV"

                                                                        # agent_M_tot_travel_time[rq_id] = 0
                                                                        # agent_F_tot_travel_time[rq_id] = 0

                                                                        agent_M_tot_travel_dist[rq_id] = 0
                                                                        agent_F_tot_travel_dist[rq_id] = 0
                                                                        #1108 Siwei: read the initial super-network according to "microtransit", "fixed route trasnit headway","virstop", and "time_period"
                                                                        T_F_network = initial_super_network["non_micro"][headway]  # the fixed route transit network remain constant - won't feedback to it
                                                                        # microtransit_run = "micro" ###**********************
                                                                        if iteration==0:
                                                                            if microtransit_run == "micro":
                                                                                if time_period in operating_periods:
                                                                                    T_micro_network = initial_super_network["micro"][headway][virstop][M_operating_hrs]
                                                                                else:
                                                                                    T_micro_network = T_F_network
                                                                            if microtransit_run == "micro_only":
                                                                                if time_period in operating_periods:
                                                                                    T_micro_network = initial_super_network["micro_only"][virstop][M_operating_hrs]
                                                                                else:
                                                                                    T_micro_network = auto_network

                                                                        else:
                                                                            if microtransit_run == "micro":
                                                                                if time_period in operating_periods:
                                                                                    T_micro_network = final_super_network["micro"][headway][virstop][M_operating_hrs][time_period][fleet_size]
                                                                                else:
                                                                                    T_micro_network = T_F_network
                                                                            if microtransit_run == "micro_only":
                                                                                if time_period in operating_periods:
                                                                                    T_micro_network = final_super_network["micro_only"][virstop][M_operating_hrs][time_period][fleet_size]
                                                                                else:
                                                                                    T_micro_network = auto_network
                                                                                # T_micro_network = initial_super_network["non_micro"][headway] #the fixed route transit network remain constant - won't feedback to it

                                                                        # *****don't calculate it along the way - only*********
                                                                        # calculate generalized cost shortest path for agent on Auto network
                                                                        if ((num_headway_scen == 1) and (num_virstop_scen == 1) and (num_fleet_size_scen == 1) and (num_operating_periods_scen == 1) and (iteration==0) and (TRPartA==False)) or ((TRPartA==True) and (test_scenario== TRPartA_test_scenario_list[0])):
                                                                            auto_visited_temp, auto_time_visited_temp, auto_dist_visited_temp, auto_fare_visited_temp,F_fare_visited_temp,M_fare_visited_temp, auto_path, auto_time_path,auto_dist_path = n_a.generalized_cost_dijsktra_OD_heap(study_area,auto_network, agent,transit_fare,microtransit_start_fare,microtransit_dist_based_rate,dt_sd_full_trnst_ntwk,PkFareFactor,OffPkFareFactor,Fixed2MicroFactor,Micro2FixedFactor,test_scenario=0, mode="C", verbose=False)
                                                                            auto_time_path_all_ = n_a.getTrajectory_O_to_D(origin, dest,auto_time_path,auto_visited_temp)
                                                                            agent_auto_visited_temp[rq_id] = auto_visited_temp
                                                                            agent_auto_time_visited_temp[rq_id] = auto_time_visited_temp
                                                                            agent_auto_dist_visited_temp[rq_id] = auto_dist_visited_temp
                                                                            agent_auto_fare_visited_temp[rq_id] = auto_fare_visited_temp
                                                                            # auto_gas_visited_temp[rq_id] = auto_gas_visited_temp
                                                                            agent_auto_path[rq_id] = auto_path
                                                                            agent_auto_time_path[rq_id] = auto_time_path
                                                                            agent_auto_time_path_all_[rq_id] = auto_time_path_all_
                                                                        else:
                                                                            auto_visited_temp = agent_auto_visited_temp[rq_id]
                                                                            auto_time_visited_temp = agent_auto_time_visited_temp[rq_id]
                                                                            auto_dist_visited_temp = agent_auto_dist_visited_temp[rq_id]
                                                                            auto_fare_visited_temp = agent_auto_fare_visited_temp[rq_id]
                                                                            auto_path = agent_auto_path[rq_id]
                                                                            auto_time_path = agent_auto_time_path[rq_id]
                                                                            auto_time_path_all_ = agent_auto_time_path_all_[rq_id]

                                                                        # if :
                                                                        # calculate generalized cost shortest path for agent on W_F network
                                                                        if ((num_virstop_scen==1) and (num_fleet_size_scen==1) and (num_operating_periods_scen==1) and (iteration==0) and (TRPartA==False)) or ((TRPartA==True) and (test_scenario== TRPartA_test_scenario_list[0])):
                                                                                visited_temp_F, time_visited_temp_F, dist_visited_temp_F, fare_visited_temp_F,F_fare_visited_temp_F,M_fare_visited_temp_F, path_F, time_path_F,dist_path_F = n_a.generalized_cost_dijsktra_OD_heap(study_area,T_F_network, agent,transit_fare, microtransit_start_fare,microtransit_dist_based_rate,dt_sd_full_trnst_ntwk,PkFareFactor,OffPkFareFactor,Fixed2MicroFactor,Micro2FixedFactor,test_scenario,mode="T",verbose=False)
                                                                                transit_time_path_all_F = n_a.getTrajectory_O_to_D(origin, dest,time_path_F,visited_temp_F)
                                                                                transit_dist_path_all_F = n_a.getTrajectory_O_to_D(origin, dest,dist_path_F,dist_visited_temp_F)  # calculate fixed transit VMT

                                                                                agent_visited_temp_F[rq_id] = visited_temp_F
                                                                                agent_time_visited_temp_F[rq_id] = time_visited_temp_F
                                                                                agent_dist_visited_temp_F[rq_id] = dist_visited_temp_F
                                                                                agent_fare_visited_temp_F[rq_id] = fare_visited_temp_F
                                                                                agent_path_F[rq_id] = path_F
                                                                                agent_time_path_F[rq_id] = time_path_F
                                                                                agent_transit_time_path_all_F[rq_id] = transit_time_path_all_F
                                                                                agent_transit_dist_path_all_F[rq_id] = transit_dist_path_all_F

                                                                        else:
                                                                            visited_temp_F = agent_visited_temp_F[rq_id]
                                                                            time_visited_temp_F = agent_time_visited_temp_F[rq_id]
                                                                            dist_visited_temp_F = agent_dist_visited_temp_F[rq_id]
                                                                            fare_visited_temp_F = agent_fare_visited_temp_F[rq_id]
                                                                            path_F = agent_path_F[rq_id]
                                                                            time_path_F = agent_time_path_F[rq_id]
                                                                            transit_time_path_all_F = agent_transit_time_path_all_F[rq_id]
                                                                            transit_dist_path_all_F = agent_transit_dist_path_all_F[rq_id]

                                                                        #calculate generalized cost shortest path for agent on W_M_F network
                                                                        if microtransit_run == "micro_only" and (time_period not in operating_periods):
                                                                            aaa=0
                                                                        else:
                                                                            visited_temp, time_visited_temp,dist_visited_temp,fare_visited_temp,F_fare_visited_temp,M_fare_visited_temp, path, time_path,dist_path = n_a.generalized_cost_dijsktra_OD_heap(study_area,T_micro_network,agent,transit_fare,microtransit_start_fare,microtransit_dist_based_rate,dt_sd_full_trnst_ntwk,PkFareFactor,OffPkFareFactor,Fixed2MicroFactor,Micro2FixedFactor,test_scenario,mode="T",verbose=False)
                                                                            transit_time_path_all_ = n_a.getTrajectory_O_to_D(origin, dest, time_path, visited_temp)
                                                                            transit_dist_path_all_ = n_a.getTrajectory_O_to_D(origin, dest, dist_path,dist_visited_temp) #calculate microtransit and fixed transit VMT

                                                                        if microtransit_run == "micro_only":
                                                                            aaa=0
                                                                            gen_cost_F = 0
                                                                            travel_time_F = 0
                                                                            travel_dist_F = 0
                                                                            agent_op_cost_F = 0
                                                                            if (time_period not in operating_periods):
                                                                                # calculate generalized cost
                                                                                gen_cost_T = 0
                                                                                # calculate travel time and distance
                                                                                travel_time_T = 0
                                                                                travel_dist_T = 0
                                                                                # calculate oout of pocket cost
                                                                                agent_op_cost_T = 0

                                                                                # agent_F_fare = F_fare_visited_temp_F[dest]
                                                                                # agent_M_fare = M_fare_visited_temp[dest]
                                                                                agent_F_fare = 0
                                                                                agent_M_fare = 0

                                                                            else:
                                                                                gen_cost_T = visited_temp[dest]
                                                                                travel_time_T = time_visited_temp[dest]
                                                                                travel_dist_T = dist_visited_temp[dest]
                                                                                agent_op_cost_T = fare_visited_temp[dest]

                                                                                agent_F_fare = 0
                                                                                agent_M_fare = M_fare_visited_temp[dest]
                                                                        else:
                                                                            # calculate generalized cost
                                                                            gen_cost_T = visited_temp[dest]
                                                                            gen_cost_F = visited_temp_F[dest]  # 1107 Siwei: add the generalized cost for without microtransit scenario
                                                                            # calculate travel time and distance
                                                                            travel_time_T = time_visited_temp[dest]
                                                                            travel_time_F = time_visited_temp_F[dest]
                                                                            travel_dist_T = dist_visited_temp[dest]
                                                                            travel_dist_F = dist_visited_temp_F[dest]

                                                                            # calculate out of pocket cost
                                                                            agent_op_cost_T = fare_visited_temp[dest]
                                                                            agent_op_cost_F = fare_visited_temp_F[dest]

                                                                            agent_F_fare = F_fare_visited_temp[dest]
                                                                            agent_M_fare = M_fare_visited_temp[dest]
                                                                        gen_cost_auto = auto_visited_temp[dest]

                                                                        # calculate travel time savings by allowing microtransit
                                                                        travel_time_auto=auto_time_visited_temp[dest]
                                                                        travel_time_delta[rq_id] = travel_time_F - travel_time_T

                                                                        travel_dist_auto = auto_dist_visited_temp[dest]
                                                                        agent_op_cost_auto=auto_fare_visited_temp[dest]
                                                                        gen_cost_delta[rq_id] = gen_cost_F - gen_cost_T
                                                                        if (gen_cost_F - gen_cost_T) < 0 and (gen_cost_F !=0):
                                                                            print("gen_cost_F", gen_cost_F, "gen_cost_T", gen_cost_T)
                                                                        #                 print("transit_time_path_all_",transit_time_path_all_,"\n","transit_time_path_all_F",transit_time_path_all_F,"\n")
                                                                        if microtransit_run == "micro_only" and (time_period not in operating_periods):
                                                                            prob_Auto, prob_T = 1,0
                                                                        else:
                                                                            prob_Auto, prob_T = b_c.binary_logit_model(gen_cost_auto, gen_cost_T)

                                                                        if iteration>0:
                                                                            diff_prob_T=abs(pre_iter_prob_T[rq_id]-prob_T)
                                                                            if diff_prob_T <= mode_change_threshold: # if probability does not change a lot, then we don't change agents' modes.
                                                                                change_mode=False
                                                                            else:
                                                                                change_mode_ag_list.append(rq_id)
                                                                                # print("change_mode agent", rq_id, "change modes", "previous_prob_T",pre_iter_prob_T[rq_id], "current_prob_T", prob_T)


                                                                        pre_iter_prob_T[rq_id]=prob_T
                                                                        if microtransit_run == "micro_only":
                                                                            prob_Auto_no_micro, prob_T_no_micro = 1,0
                                                                        else:
                                                                            prob_Auto_no_micro, prob_T_no_micro = b_c.binary_logit_model(gen_cost_auto, gen_cost_F)

                                                                        # calculate the mobility logsum change (logsum delta)
                                                                        mob_logsum_micro = b_c.mobility_logsum(gen_cost_auto, gen_cost_T)
                                                                        mob_logsum_fixed = b_c.mobility_logsum(gen_cost_auto, gen_cost_F)
                                                                        mob_logsum_delta[rq_id] = (mob_logsum_micro - mob_logsum_fixed)/agent_beta_fare[rq_id]

                                                                        # agent_choice_prob[rq_id] = [prob_Auto, prob_T] #07/04/2024 Siwei: move to "change_mode=True"
                                                                        off_micro_agent_choice_prob[rq_id]= [prob_Auto_no_micro, prob_T_no_micro]
                                                                        iter_agent_choice_prob[iteration][rq_id] = [prob_Auto, prob_T]


                                                                        ran_num = random.random()
                                                                        # ran_num = np.random.gumbel()
                                                                        # agent_ran_num[rq_id]=ran_num #07/04/2024 Siwei: move to "change_mode=True"
                                                                        #             print("random number:",ran_num)
                                                                        #calculate the output metrics of turn-off microtransit
                                                                        if (num_virstop_scen == 1) and (num_fleet_size_scen == 1) and (num_operating_periods_scen == 1) and (iteration==0):
                                                                            off_micro_ag_tot_dist[rq_id] = 0
                                                                            off_micro_ag_tot_tt[rq_id] = 0
                                                                            off_micro_agent_gen_cost[rq_id] = 0
                                                                            off_micro_ag_VMT_W[rq_id] = 0
                                                                            off_micro_agent_F_tot_wait_time[rq_id] = 0
                                                                            off_micro_agent_tot_walk_time[rq_id] = 0

                                                                            off_micro_agent_ran_num[rq_id]=ran_num
                                                                            if ran_num < prob_T_no_micro:

                                                                                off_micro_ag_mode[rq_id] = "T"
                                                                                # off_micro_c_tt_travel_time[rq_id] = 0
                                                                                off_micro_T_trips += 1
                                                                                off_micro_ag_tot_tt[rq_id] = travel_time_F
                                                                                # off_micro_f_tt_travel_time[rq_id] = travel_time_F
                                                                                off_micro_ag_tot_dist[rq_id] = travel_dist_F/mile_meter
                                                                                # off_micro_ag_VMT_F[rq_id] = off_micro_ag_dist_F

                                                                                off_micro_agent_gen_cost[rq_id]=gen_cost_F
                                                                                trimmed_transit_time_path_F = transit_time_path_all_F[1:-1]
                                                                                pre_link_type = None
                                                                                # request_time=test_agent.rq_time
                                                                                sum_link_type_F = 0  # recorde the pure walking trips
                                                                                # 11/07 Siwei: record the time of reaching the previous link
                                                                                pre_time = 0

                                                                                # 1107 Siwei
                                                                                # agent_M_tot_wait_time[rq_id] = 0
                                                                                # agent_F_tot_wait_time[rq_id] = 0


                                                                                # agent_travel_time[rq_id] = travel_time_T
                                                                                # agent_travel_dist[rq_id] = travel_dist_T
                                                                                # agent_op_cost[rq_id] = agent_op_cost_T  # agent_out-of-pocket cost
                                                                                off_micro_agent_op_cost[rq_id] = agent_op_cost_F
                                                                                for (node, time, link_type,route) in trimmed_transit_time_path_F:
                                                                                    # calculate the number of fixed route transit trip legs
                                                                                    if (pre_link_type != 1 and pre_link_type != 3) and (link_type == 1):
                                                                                        F_origin_node = node
                                                                                        request_time = rq_time + int(time)
                                                                                    if (pre_link_type == 1) and (link_type != 1 and link_type != 3):
                                                                                        F_dest_node = pre_node
                                                                                        off_micro_F_trips += 1

                                                                                    # 11/07 Siwei: calculate the fixed route transit waiting time
                                                                                    if (pre_link_type != 2) and (link_type == 2):
                                                                                        F_wait_time = (time - pre_time)
                                                                                        off_micro_agent_F_tot_wait_time[rq_id] += F_wait_time
                                                                                    # 11/07 Siwei: calculate the walking time
                                                                                    if link_type == 0:
                                                                                        walk_time = (time - pre_time)
                                                                                        off_micro_agent_tot_walk_time[rq_id] += walk_time


                                                                                    pre_time = time
                                                                                    sum_link_type_F += link_type
                                                                                    pre_node = node
                                                                                    pre_link_type = link_type
                                                                                if sum_link_type_F == 0:
                                                                                    off_micro_W_trips += 1

                                                                                #calculate VMT for fixed transit and walking when microtransit turned off
                                                                                trimmed_transit_dist_path_F=transit_dist_path_all_F[1:-1]
                                                                                pre_dist = 0
                                                                                pre_node = transit_dist_path_all_[0]
                                                                                for (node, dist, link_type,route) in trimmed_transit_dist_path_F:
                                                                                    # calculate the number of fixed route transit trip legs
                                                                                    if link_type == 1:
                                                                                        distance_F = dist - pre_dist
                                                                                        off_micro_VMT_F+=(distance_F/mile_meter)
                                                                                        off_micro_transit_link_vmt[time_period][(pre_node, node)] += (distance_F / mile_meter)
                                                                                        off_micro_transit_link_pax[time_period][(pre_node, node)]+=1
                                                                                    if link_type==0:
                                                                                        distance_W = dist - pre_dist
                                                                                        off_micro_VMT_W += (distance_W / mile_meter)
                                                                                        off_micro_ag_VMT_W[rq_id]+=(distance_W / mile_meter)
                                                                                    pre_dist = dist
                                                                                    pre_node = node


                                                                            else:  # agent choose the car mode
                                                                                off_micro_ag_mode[rq_id] = "C"
                                                                                off_micro_ag_tot_tt[rq_id]= travel_time_auto
                                                                                # off_micro_c_tt_travel_time[rq_id]= travel_time_auto
                                                                                # off_micro_f_tt_travel_time[rq_id]= 0
                                                                                off_micro_ag_tot_dist[rq_id] = travel_dist_auto/mile_meter
                                                                                off_micro_auto_trips+=1
                                                                                # agent_travel_time[rq_id] = travel_time_auto
                                                                                # agent_travel_dist[rq_id] = travel_dist_auto
                                                                                off_micro_agent_op_cost[rq_id] = agent_op_cost_auto  # agent_out-of-pocket
                                                                                off_micro_VMT_auto += (travel_dist_auto/mile_meter)
                                                                                off_micro_agent_gen_cost[rq_id]=gen_cost_auto



                                                                                # off_micro_auto_trips+=1
                                                                        #turn on microtransit
                                                                        if change_mode==True:

                                                                            if iteration ==0: #when it is in iteration 0, all travelers get M and F travel time initialized
                                                                                agent_switch_mode[rq_id] = 0
                                                                                agent_M_tot_travel_time[rq_id] = 0
                                                                                agent_F_tot_travel_time[rq_id] = 0
                                                                                ag_VMT_F[rq_id] = 0
                                                                                ag_VMT_W[rq_id] = 0
                                                                                ag_VMT_M[rq_id] = 0
                                                                                ag_num_W_trips[rq_id] = 0
                                                                                ag_num_M_trips[rq_id] = 0
                                                                                ag_num_F_trips[rq_id] = 0

                                                                            agent_ran_num[rq_id] = ran_num
                                                                            agent_choice_prob[rq_id] = [prob_Auto,prob_T]

                                                                            if ran_num < prob_T:  # agent choose the transit mode
                                                                                ##when mode change=true, and transit mode is chosen, all travelers get M and F travel time initialized
                                                                                current_mode = "T"
                                                                                pre_iter_mode[rq_id] = current_mode

                                                                                agent_switch_mode[rq_id] = 0
                                                                                agent_M_tot_travel_time[rq_id] = 0
                                                                                agent_F_tot_travel_time[rq_id] = 0
                                                                                ag_VMT_F[rq_id] = 0
                                                                                ag_VMT_W[rq_id] = 0
                                                                                ag_VMT_M[rq_id] = 0
                                                                                ag_num_W_trips[rq_id] = 0
                                                                                ag_num_M_trips[rq_id] = 0
                                                                                ag_num_F_trips[rq_id] = 0



                                                                                agent_gen_cost[rq_id] = gen_cost_T
                                                                                if ran_num>=prob_T_no_micro:
                                                                                    agent_switch_mode[rq_id]=1 #this means: if no microtransit, this agent will choose car, but with microtransit, this agent will choose transit mode
                                                                                trimmed_transit_time_path = transit_time_path_all_[1:-1]
                                                                                pre_link_type = None
                                                                                # request_time=test_agent.rq_time
                                                                                sum_link_type = 0  # recorde the pure walking trips
                                                                                # 11/07 Siwei: record the time of reaching the previous link
                                                                                pre_time = 0
                                                                                M_tot_wait_time = 0

                                                                                # 1107 Siwei
                                                                                agent_M_tot_wait_time[rq_id] = 0
                                                                                agent_F_tot_wait_time[rq_id] = 0



                                                                                agent_tot_walk_time[rq_id] = 0
                                                                                agent_travel_time[rq_id]=travel_time_T
                                                                                agent_travel_dist[rq_id] =travel_dist_T
                                                                                agent_op_cost[rq_id] = agent_op_cost_T  # agent_out-of-pocket cost

                                                                                agent_op_cost_Fix[rq_id] = agent_F_fare
                                                                                agent_op_cost_Micro[rq_id] = agent_M_fare

                                                                                for (node, time, link_type,route) in trimmed_transit_time_path:
                                                                                    # calculate the number of microtransit trip legs
                                                                                    if (pre_link_type != 4) and (link_type == 4): ##1228 testing!
                                                                                    # if (pre_link_type == 5) and (link_type == 4):  ##1228 testing!
                                                                                        M_origin_node = node
                                                                                        request_time = rq_time + round(time,0)
                                                                                    if (pre_link_type == 4) and (link_type != 4): ##1228 testing!
                                                                                    # if (pre_link_type == 4) and (link_type == 5):  ##1228 testing!
                                                                                        M_dest_node = pre_node
                                                                                        # MicroTransitNodes
                                                                                        rq_id_fleetpy = rq_id
                                                                                        while (rq_id_fleetpy in M_rq_id_list): #this condition is to deal with the situation where 1 person has more than 1 microtransit trips
                                                                                            rq_id_fleetpy += 1

                                                                                        try:
                                                                                            writer_M.writerow({'rq_time': request_time, 'start': MicroTransitNodes[M_origin_node],'end': MicroTransitNodes[M_dest_node], 'request_id': rq_id_fleetpy})
                                                                                        except:
                                                                                            raise Exception("M_origin_node",M_origin_node,"M_dest_node",M_dest_node,"pre_link_type",pre_link_type,"link_type",link_type,"node",node,"pre_node",pre_node,"trimmed_transit_time_path",trimmed_transit_time_path)
                                                                                        M_rq_id_list.append(rq_id_fleetpy)
                                                                                        #document the previous iteration microtransit information
                                                                                        #rq_time, start, end, request_id
                                                                                        pre_iter_M_info[rq_id]=(request_time,MicroTransitNodes[M_origin_node],MicroTransitNodes[M_dest_node],rq_id_fleetpy)

                                                                                        num_M_trips += 1
                                                                                        ag_num_M_trips[rq_id] +=1
                                                                                    #                         print("depart",rq_time,"O",M_origin_node,"D",M_dest_node,"request_time",request_time,"walking_time",request_time-rq_time)
                                                                                    # calculate the number of fixed route transit trip legs
                                                                                    if (pre_link_type != 1 and pre_link_type != 3) and (link_type == 1):
                                                                                        F_origin_node = node
                                                                                        request_time = rq_time + int(time)
                                                                                    if (pre_link_type == 1) and (link_type != 1 and link_type != 3):
                                                                                        F_dest_node = pre_node
                                                                                        num_F_trips += 1
                                                                                        ag_num_F_trips[rq_id] +=1
                                                                                        writer_F.writerow({'depart_time':request_time , 'start': F_origin_node,'end': F_dest_node,'request_id': rq_id})
                                                                                        # document the previous iteration fixed route transit information
                                                                                        # rq_time, start, end, request_id
                                                                                        pre_iter_F_info[rq_id]=(request_time,F_origin_node,F_dest_node,rq_id)
                                                                                    # 11/07 Siwei: calculate the microtransit waiting time
                                                                                    if (pre_link_type != 5) and (link_type == 5):
                                                                                        M_wait_time = (time - pre_time)
                                                                                        agent_M_tot_wait_time[rq_id] += M_wait_time

                                                                                    # 11/07 Siwei: calculate the fixed route transit waiting time
                                                                                    if (pre_link_type != 2) and (link_type == 2):
                                                                                        F_wait_time = (time - pre_time)
                                                                                        agent_F_tot_wait_time[rq_id] += F_wait_time

                                                                                    # 11/07 Siwei: calculate the walking time
                                                                                    if link_type == 0:
                                                                                        walk_time = (time - pre_time)
                                                                                        agent_tot_walk_time[rq_id] += walk_time
                                                                                    # 12/22 Siwei: calculate the microtransit travel time
                                                                                    if link_type ==4:
                                                                                        M_travel_time = (time - pre_time)
                                                                                        agent_M_tot_travel_time[rq_id] += M_travel_time

                                                                                    # 12/22 Siwei: calculate the fixed route transit travel time
                                                                                    if link_type == 1 or link_type ==3:
                                                                                        F_travel_time = (time - pre_time)
                                                                                        agent_F_tot_travel_time[rq_id] += F_travel_time

                                                                                    pre_time = time
                                                                                    sum_link_type += link_type
                                                                                    pre_node = node
                                                                                    pre_link_type = link_type
                                                                                if sum_link_type == 0:
                                                                                    num_W_trips += 1
                                                                                    ag_num_W_trips[rq_id]+=1
                                                                                #calculate pure microtransit users:
                                                                                if ag_num_M_trips[rq_id] != 0 and ag_num_F_trips[rq_id] == 0:
                                                                                    pure_M_user += 1
                                                                                # calculate pure fixed route transit users:
                                                                                if ag_num_M_trips[rq_id] == 0 and ag_num_F_trips[rq_id] != 0:
                                                                                    pure_F_user += 1
                                                                                if ag_num_M_trips[rq_id] != 0 and ag_num_F_trips[rq_id] != 0:
                                                                                    M_F_user += 1
                                                                                # calculate VMT for fixed transit and walking when microtransit turned on


                                                                                trimmed_transit_dist_path = transit_dist_path_all_[1:-1]
                                                                                pre_dist = 0
                                                                                pre_node = transit_dist_path_all_[0]
                                                                                for (node, dist, link_type,route) in trimmed_transit_dist_path:
                                                                                    # calculate the number of fixed route transit trip legs
                                                                                    if link_type == 1:
                                                                                        distance_F = dist - pre_dist
                                                                                        VMT_F += (distance_F / mile_meter)
                                                                                        ag_VMT_F[rq_id] += (distance_F / mile_meter)
                                                                                        transit_link_vmt[time_period][(pre_node, node)]+=(distance_F / mile_meter)
                                                                                        transit_link_pax[time_period][(pre_node, node)]+=1



                                                                                    if link_type == 0:
                                                                                        distance_W = dist - pre_dist
                                                                                        VMT_W += (distance_W / mile_meter)
                                                                                        ag_VMT_W[rq_id] += (distance_W / mile_meter)

                                                                                    if link_type == 4:
                                                                                        distance_M = dist - pre_dist
                                                                                        VMT_M += (distance_M / mile_meter)
                                                                                        ag_VMT_M[rq_id] += (distance_M / mile_meter)
                                                                                    pre_dist = dist
                                                                                    pre_node=node

                                                                            else:  # agent choose the car mode

                                                                                current_mode = "C"
                                                                                if (iteration>=1):
                                                                                    if pre_iter_mode[rq_id] != current_mode:
                                                                                        agent_M_tot_travel_time[rq_id] = 0
                                                                                        agent_F_tot_travel_time[rq_id] = 0

                                                                                pre_iter_mode[rq_id] = current_mode
                                                                                num_car_trips += 1
                                                                                agent_travel_time[rq_id] = travel_time_auto
                                                                                agent_travel_dist[rq_id] = travel_dist_auto
                                                                                agent_op_cost[rq_id] = agent_op_cost_auto  # agent_out-of-pocket cost
                                                                                agent_op_cost_Fix[rq_id] = 0
                                                                                agent_op_cost_Micro[rq_id] = 0
                                                                                VMT_auto+=(travel_dist_auto/mile_meter)
                                                                                agent_gen_cost[rq_id]=gen_cost_auto

                                                                                # agent_M_tot_travel_time[rq_id] = 0
                                                                                # agent_F_tot_travel_time[rq_id] = 0
                                                                                # ag_VMT_F[rq_id] = 0
                                                                                # ag_VMT_W[rq_id] = 0
                                                                                # ag_VMT_M[rq_id] = 0
                                                                                # ag_num_W_trips[rq_id] = 0
                                                                                # ag_num_M_trips[rq_id] = 0
                                                                                # ag_num_F_trips[rq_id] = 0
                                                                                # # agent_M_tot_wait_time[rq_id] = 0
                                                                                # agent_F_tot_wait_time[rq_id] = 0
                                                                                # agent_tot_walk_time[rq_id] = 0
                                                                        else:
                                                                            ag_prev_iter_mode=pre_iter_mode[rq_id]
                                                                            if ag_prev_iter_mode =="T":

                                                                                VMT_F += ag_VMT_F[rq_id]
                                                                                VMT_W += ag_VMT_W[rq_id]
                                                                                VMT_M += ag_VMT_M[rq_id]

                                                                                num_W_trips += ag_num_W_trips[rq_id]
                                                                                num_F_trips += ag_num_F_trips[rq_id]
                                                                                num_M_trips += ag_num_M_trips[rq_id]

                                                                                if ag_num_M_trips[rq_id] != 0 and ag_num_F_trips[rq_id] == 0:
                                                                                    pure_M_user += 1
                                                                                # calculate pure fixed route transit users:
                                                                                if ag_num_M_trips[rq_id] == 0 and ag_num_F_trips[rq_id] != 0:
                                                                                    pure_F_user += 1
                                                                                if ag_num_M_trips[rq_id] != 0 and ag_num_F_trips[rq_id] != 0:
                                                                                    M_F_user += 1

                                                                                if rq_id in pre_iter_M_info:
                                                                                    # rq_time, start, end, request_id
                                                                                    (pre_ier_rq_time, pre_iter_start,pre_iter_end, pre_iter_request_id) = pre_iter_M_info[rq_id]
                                                                                    rq_id_fleetpy = pre_iter_request_id
                                                                                    while (rq_id_fleetpy in M_rq_id_list):  # this condition is to deal with the situation where 1 person has more than 1 microtransit trips
                                                                                        rq_id_fleetpy += 1
                                                                                    writer_M.writerow({'rq_time': pre_ier_rq_time,'start': pre_iter_start,'end': pre_iter_end,'request_id':rq_id_fleetpy})
                                                                                    M_rq_id_list.append(rq_id_fleetpy)
                                                                                if rq_id in pre_iter_F_info:
                                                                                    #(request_time,F_origin_node,F_dest_node,rq_id)
                                                                                    (pre_ier_rq_time_F, pre_iter_start_F,pre_iter_end_F, pre_iter_request_id_F) = pre_iter_F_info[rq_id]
                                                                                    writer_F.writerow({'depart_time': pre_ier_rq_time_F, 'start': pre_iter_start_F,'end': pre_iter_end_F, 'request_id': pre_iter_request_id_F})
                                                                                    # writer_F.writerow({'rq_time': pre_ier_rq_time_F,'start': pre_iter_start_F,'end': pre_iter_end_F,'request_id': pre_iter_request_id_F})


                                                                            else:
                                                                                num_car_trips += 1
                                                                                VMT_auto += (travel_dist_auto / mile_meter)


                                                                        #         print("prob_Auto",prob_Auto,"prob_T",prob_T)

                                                                    num_transit_users = tt_num_agents - num_car_trips

                                                                    print("number of agents prob_T changed greater than threshold",mode_change_threshold,":", len(change_mode_ag_list))

                                                                    # print("assign pro iteration",iteration)
                                                                    # iter_agent_choice_prob[iteration] = agent_choice_prob
                                                                    # print("assign pro iteration", iteration,"iter_agent_choice_prob[iteration]",iter_agent_choice_prob[iteration])

                                                                csvfile_M.close()

                                                            csvfile_F.close()


                                                            ####################
                                                            #Convergence test right after the Binary Choice Model
                                                            ##########################################
                                                            # aggregated level convergence test
                                                            if indiv_converg_test == False:
                                                                if iteration >= 1:
                                                                    print("convergence check.....")
                                                                    # converged,per_diff=conv_test.convergence_test(M_share,M_share_pre)
                                                                    # print("Aggregated Level Convergence Test","M_share",M_share,"M_share_pre",M_share_pre,"per_diff",per_diff,"converged",converged)
                                                            else:
                                                                if iteration >= 1:
                                                                    print("convergence check.....")
                                                                    # for rq_id in change_mode_ag_list:
                                                                    #     ag_prob_cur_iter=iter_agent_choice_prob[iteration][rq_id][1]
                                                                    #     ag_prob_pre_iter = iter_agent_choice_prob[iteration-1][rq_id][1]
                                                                    #     print("iteration",iteration,"pro_change_agent",rq_id,"pre_iter_prob",ag_prob_pre_iter,"cur_iter_prob",ag_prob_cur_iter)
                                                                    # print("*******within convergence test**********","current iteration", iteration,"current iter_agent_choice_prob",iter_agent_choice_prob[iteration])
                                                                    # print("*******within convergence test**********", "previous iteration",iteration-1, "current iter_agent_choice_prob",iter_agent_choice_prob[iteration-1])
                                                                    converged_, sum_sq_per_diff = conv_test.indiv_convergence_test(iter_agent_choice_prob[iteration], iter_agent_choice_prob[iteration - 1],convergence_gap)

                                                                    iter_sum_sq_per_diff.append((iteration,sum_sq_per_diff))

                                                                    if converged_ == True:
                                                                        num_converged += 1
                                                                        if num_converged == 2:
                                                                            converged = True


                                                                    if iteration == maximum_iteration:
                                                                        #if it is iteration_debug mode, then it is 15 iterations;
                                                                        #if it is normal mode, then it is 10 iterations;
                                                                        converged = True

                                                                    print("Individual Level Convergence Test:","sum of square percentage difference ", sum_sq_per_diff, "converged",converged)
                                                                    if converged == True:
                                                                        F_oper_cost = 0
                                                                        if microtransit_run == "micro_only":
                                                                            transit_tt_VMT=0
                                                                        else:
                                                                            for (route_id,dist,num_stops) in transit_line_dist_list:
                                                                                F_oper_cost += 2 * (F_operating_hrs * 60 / headway) * transit_line_duration[route_id] / 60 * 170  #5 lines and $fixed route transit operating cost 170/veh-hour
                                                                                transit_tt_VMT += 2 * (F_operating_hrs * 60 / headway) * dist

                                                                        F_gas_cost = transit_tt_VMT * F_gas_per_mile
                                                                        F_oper_cost += F_gas_cost

                                                                        M_mode_operating_cost = M_operating_hrs * fleet_size * 130  # microtransit operating cost $130/veh-hour
                                                                        output_metrics.write_fixed_transit_link_vmt(study_area,dt_sd_full_trnst_ntwk,fleet_size,M_operating_hrs,headway,virstop,transit_link_vmt,transit_link_pax,TRPartA,BayesianOptimization,test_scenario,debug_mode,microtransit="micro")
                                                                        output_metrics.write_fixed_transit_link_vmt(study_area,dt_sd_full_trnst_ntwk,fleet_size,M_operating_hrs,headway,virstop,off_micro_transit_link_vmt,off_micro_transit_link_pax,TRPartA,BayesianOptimization,test_scenario,debug_mode,microtransit="non_micro")

                                                                        avg_period_total_weighted_5_min,avg_period_total_weighted_10_min,avg_period_total_weighted_15_min = output_metrics.all_scenario_micro_acc_calculation(microtransit_run,headway, virstop,M_operating_hrs,fleet_size,study_area,debug_mode,dt_sd_full_trnst_ntwk,zonal_partition,TRPartA,BayesianOptimization,test_scenario) #1112 Siwei: comment to make the program run faster.


                                                                        print("The program is converged.")

                                                                        # individual_result_dir = os.path.join(output_folder,"individual_results.csv")
                                                                        if debug_mode==True:

                                                                            individual_result_dir = os.path.join(output_folder,"debug_individual_results_%s_M_fsize_%s_op_hr_%s_hw_%s_virstop_%s_scen_%s.csv" % (str(microtransit_run),str(fleet_size),str(M_operating_hrs), str(headway), str(virstop),str(test_scenario)))
                                                                        else:

                                                                            individual_result_dir=os.path.join(output_folder,"individual_results_%s_M_fsize_%s_op_hr_%s_hw_%s_virstop_%s_scen_%s.csv" % (str(microtransit_run),str(fleet_size),str(M_operating_hrs), str(headway),str(virstop),str(test_scenario)))



                                                                        with open(individual_result_dir, 'w+', newline='') as csvfile_indiv:
                                                                            if TRPartA==True:
                                                                                fieldnames = ["rq_id", "origin", "dest","dp_time", "mode","switch_mode", "tt (s)","dist (mi)", "tt_C (s)",
                                                                                              "dist_C (mi)", "tt_M (s)","dist_M (mi)", "tt_F (s)","dist_F (mi)", "tt_W (s)",
                                                                                              "dist_W (mi)","agent_M_tot_wait_time (s)","agent_F_tot_wait_time (s)","out_of_pock_cost ($)","tt_saved_with_micro (s)",
                                                                                              "gen_cost_saved_with_micro","mob_logsum_increased_with_micro",
                                                                                              "agent_prob (auto,transit)","agent_ran_num","income (USD)","agent_transit_15min_acc"]

                                                                            else:
                                                                                fieldnames = ["rq_id","origin","dest","dp_time", "mode","switch_mode","tt (s)","dist (mi)","tt_C (s)","dist_C (mi)","tt_M (s)","dist_M (mi)","tt_F (s)","dist_F (mi)","tt_W (s)","dist_W (mi)","agent_M_tot_wait_time (s)", "agent_F_tot_wait_time (s)","out_of_pock_cost ($)", "tt_saved_with_micro (s)", "gen_cost_saved_with_micro","mob_logsum_increased_with_micro", "agent_prob (auto,transit)","agent_ran_num"]
                                                                            writer_indiv = csv.DictWriter(csvfile_indiv, fieldnames=fieldnames)
                                                                            writer_indiv.writeheader()

                                                                            total_op_cost=0
                                                                            total_T_revenue = 0
                                                                            total_F_revenue = 0
                                                                            total_M_revenue = 0
                                                                            F_sub_per_ridership=0
                                                                            M_sub_per_ridership=0
                                                                            tt_sub_per_ridership=0
                                                                            M_revenue = 0
                                                                            F_revenue =0
                                                                            total_mob_increase=0
                                                                            total_switch_mode=0
                                                                            total_M_wait_time=0
                                                                            total_F_wait_time = 0
                                                                            total_walk_time = 0
                                                                            tt_gen_cost = 0 #total generalized cost for the selected mode
                                                                            num_M=0
                                                                            num_F=0
                                                                            num_W=0
                                                                            for rq_id in travel_time_delta:
                                                                                tt_saved_with_micro = 0
                                                                                gen_cost_saved_with_micro = 0
                                                                                c_tt_travel_time = 0
                                                                                c_tt_travel_dist = 0
                                                                                # mob_logsum_increased_with_micro = 0
                                                                                if rq_id in agent_M_tot_wait_time:
                                                                                    mode="T"
                                                                                    m_waiting_time = agent_M_tot_wait_time[rq_id]
                                                                                    if m_waiting_time>0:
                                                                                        num_M+=1
                                                                                    f_waiting_time = agent_F_tot_wait_time[rq_id]
                                                                                    if f_waiting_time > 0:
                                                                                        num_F += 1
                                                                                    T_walking_time=agent_tot_walk_time[rq_id]
                                                                                    if T_walking_time > 0:
                                                                                        num_W += 1

                                                                                    total_T_revenue += agent_op_cost[rq_id]
                                                                                    total_F_revenue += agent_op_cost_Fix[rq_id]
                                                                                    total_M_revenue += agent_op_cost_Micro[rq_id]
                                                                                    if m_waiting_time >0:
                                                                                        tt_saved_with_micro = travel_time_delta[rq_id]
                                                                                        gen_cost_saved_with_micro = gen_cost_delta[rq_id]
                                                                                        # mob_logsum_increased_with_micro = mob_logsum_delta[rq_id]
                                                                                        # c_tt_travel_time = 0
                                                                                        # c_tt_travel_dist = 0
                                                                                else:
                                                                                    mode="C"
                                                                                    m_waiting_time = 0
                                                                                    f_waiting_time = 0
                                                                                    T_walking_time = 0
                                                                                    c_tt_travel_time = agent_travel_time[rq_id]
                                                                                    c_tt_travel_dist = agent_travel_dist[rq_id]/mile_meter

                                                                                origin=agent_O[rq_id]
                                                                                dest=agent_D[rq_id]
                                                                                dp_time=agent_rq_time[rq_id]

                                                                                total_op_cost+=agent_op_cost[rq_id]
                                                                                total_mob_increase+=mob_logsum_delta[rq_id]
                                                                                total_switch_mode+=agent_switch_mode[rq_id]
                                                                                total_M_wait_time += m_waiting_time
                                                                                total_F_wait_time += f_waiting_time
                                                                                total_walk_time += T_walking_time
                                                                                tt_gen_cost += agent_gen_cost[rq_id]

                                                                                # agent_F_fare = OrderedDict()  # agent fixed route transit fare cost
                                                                                # agent_M_fare = OrderedDict()  # agent micro transit fare cost
                                                                                if TRPartA == True:
                                                                                    writer_indiv.writerow({"rq_id": rq_id,"origin": origin, "dest": dest,
                                                                                         "dp_time": dp_time,"mode": mode, "switch_mode":agent_switch_mode[rq_id],
                                                                                         "tt (s)": agent_travel_time[rq_id],"dist (mi)": agent_travel_dist[rq_id] / mile_meter,
                                                                                         "tt_C (s)": c_tt_travel_time,"dist_C (mi)": c_tt_travel_dist,"tt_M (s)":agent_M_tot_travel_time[rq_id],
                                                                                         "dist_M (mi)": ag_VMT_M[rq_id],"tt_F (s)":agent_F_tot_travel_time[rq_id],"dist_F (mi)": ag_VMT_F[rq_id],
                                                                                          "agent_M_tot_wait_time (s)": m_waiting_time,"agent_F_tot_wait_time (s)": f_waiting_time,
                                                                                         "out_of_pock_cost ($)":agent_op_cost[rq_id],"tt_saved_with_micro (s)": tt_saved_with_micro,"gen_cost_saved_with_micro": gen_cost_delta[rq_id],
                                                                                         "mob_logsum_increased_with_micro":mob_logsum_delta[rq_id],"agent_prob (auto,transit)":agent_choice_prob[rq_id],
                                                                                         "agent_ran_num": agent_ran_num[rq_id],
                                                                                         "tt_W (s)": T_walking_time,"dist_W (mi)": ag_VMT_W[rq_id],"income (USD)":agent_income[rq_id],"agent_transit_15min_acc":agent_transit_15min_acc[rq_id]})
                                                                                else:
                                                                                    writer_indiv.writerow({"rq_id":rq_id, "origin":origin, "dest":dest, "dp_time":dp_time,
                                                                                                 "mode":mode,"switch_mode":agent_switch_mode[rq_id], "tt (s)":agent_travel_time[rq_id],"dist (mi)":agent_travel_dist[rq_id]/mile_meter,
                                                                                                   "tt_C (s)":c_tt_travel_time, "dist_C (mi)":c_tt_travel_dist,
                                                                                                   "tt_M (s)":agent_M_tot_travel_time[rq_id], "dist_M (mi)":ag_VMT_M[rq_id], "tt_F (s)":agent_F_tot_travel_time[rq_id],"dist_F (mi)":ag_VMT_F[rq_id],
                                                                                                  "agent_M_tot_wait_time (s)": m_waiting_time,"agent_F_tot_wait_time (s)": f_waiting_time,
                                                                                                 "out_of_pock_cost ($)":agent_op_cost[rq_id] ,
                                                                                                 "tt_saved_with_micro (s)":tt_saved_with_micro,
                                                                                                 "gen_cost_saved_with_micro":gen_cost_delta[rq_id],"mob_logsum_increased_with_micro":mob_logsum_delta[rq_id],
                                                                                                 "agent_prob (auto,transit)":agent_choice_prob[rq_id],
                                                                                                "agent_ran_num":agent_ran_num[rq_id],
                                                                                                "tt_W (s)":T_walking_time,"dist_W (mi)":ag_VMT_W[rq_id]})



                                                                        csvfile_indiv.close()
                                                                        if num_M!=0:
                                                                            # avg_M_wait_time=total_M_wait_time / num_M
                                                                            avg_M_wait_time = total_M_wait_time /num_M_trips #total microtransit waiting time/microtransit trips
                                                                        else:
                                                                            avg_M_wait_time=0
                                                                        if num_F!=0:
                                                                            avg_F_wait_time = total_F_wait_time/num_F_trips #total waiting time/ total number of F trips
                                                                        else:
                                                                            avg_F_wait_time=0
                                                                        if num_W!=0:
                                                                            avg_walk_time = total_walk_time/num_W
                                                                        else:
                                                                            avg_walk_time=0
                                                                        # F_revenue=num_F_trips*transit_fare
                                                                        F_revenue=total_F_revenue
                                                                        M_revenue=total_M_revenue
                                                                        if num_F_trips==0:
                                                                            if microtransit_run == "micro_only":
                                                                                F_sub_per_ridership = 0
                                                                            else:
                                                                                F_sub_per_ridership=-999
                                                                        else:
                                                                            F_sub_per_ridership=(F_oper_cost-F_revenue)/num_F_trips # fixed transit subsidy per fixed transit trip
                                                                        # M_sub_per_ridership=(M_mode_operating_cost-M_revenue)/num_M_trips
                                                                        # tt_sub=(F_oper_cost+M_mode_operating_cost-F_revenue-M_revenue)
                                                                        # tt_sub_per_ridership=tt_sub/(num_F_trips+num_M_trips)


                                                                        #calculate the turn-off micro transit scenario
                                                                        #under every different fixed route transit headway, we need to calculate the cost again
                                                                        if (num_virstop_scen==1) and (num_fleet_size_scen==1) and (num_operating_periods_scen==1) and (microtransit_run=="micro") and (repositioning==True):
                                                                            #calculate off_micro metrics
                                                                            off_micro_F_revenue=off_micro_F_trips*transit_fare
                                                                            if off_micro_F_trips==0:
                                                                                off_micro_F_sub_per_ridership=-999
                                                                                off_micro_tt_sub = -999
                                                                            else:
                                                                                off_micro_tt_sub =(F_oper_cost - off_micro_F_revenue)
                                                                                off_micro_F_sub_per_ridership = off_micro_tt_sub / off_micro_F_trips

                                                                            off_micro_w_trip_legs=0
                                                                            off_micro_tot_walk_time=0
                                                                            off_micro_avg_walk_time=0
                                                                            off_micro_F_trip_legs=0
                                                                            off_micro_tot_F_wait_time=0
                                                                            off_micro_avg_F_wait_time=0
                                                                            off_micro_total_op_cost=0
                                                                            off_micro_tt_gen_cost = 0

                                                                            for rq_id in travel_time_delta:
                                                                                off_micro_total_op_cost += off_micro_agent_op_cost[rq_id]
                                                                                off_micro_tt_gen_cost += off_micro_agent_gen_cost[rq_id]

                                                                            for rq_id in off_micro_agent_tot_walk_time:
                                                                                # off_micro_total_op_cost+=off_micro_agent_op_cost[rq_id]
                                                                                if off_micro_agent_tot_walk_time[rq_id]>0:
                                                                                    off_micro_w_trip_legs+=1
                                                                                    off_micro_tot_walk_time+=off_micro_agent_tot_walk_time[rq_id]
                                                                                if off_micro_agent_F_tot_wait_time[rq_id]>0:
                                                                                    off_micro_F_trip_legs+=1
                                                                                    off_micro_tot_F_wait_time+=off_micro_agent_F_tot_wait_time[rq_id]



                                                                            if microtransit_run =="micro_only":
                                                                                off_micro_avg_walk_time = 0
                                                                            else:
                                                                                off_micro_avg_walk_time = off_micro_tot_walk_time/off_micro_w_trip_legs

                                                                            if microtransit_run == "micro_only":
                                                                                off_micro_avg_F_wait_time = 0
                                                                            else:
                                                                                if off_micro_F_trip_legs==0:
                                                                                    off_micro_avg_F_wait_time = headway*60/2
                                                                                else:
                                                                                    # off_micro_avg_F_wait_time=off_micro_tot_F_wait_time/off_micro_F_trip_legs
                                                                                    off_micro_avg_F_wait_time=off_micro_tot_F_wait_time/off_micro_F_trips # total waiting time/fixed transit trips
                                                                            off_micro_tt_VMT = off_micro_VMT_auto + 0+ transit_tt_VMT

                                                                            # microtransit_run="non_micro"
                                                                            microtransit_run_pre=microtransit_run
                                                                            microtransit_run="non_micro"
                                                                            if debug_mode == True:
                                                                                individual_result_dir = os.path.join(output_folder,"debug_individual_results_%s_M_fsize_%s_op_hr_%s_hw_%s_virstop_%s_scen_%s.csv" % (str(microtransit_run), str(fleet_size),str(M_operating_hrs), str(headway),str(virstop),str(test_scenario)))
                                                                            else:
                                                                                individual_result_dir = os.path.join(output_folder,"individual_results_%s_M_fsize_%s_op_hr_%s_hw_%s_virstop_%s_scen_%s.csv" % (str(microtransit_run), str(fleet_size),str(M_operating_hrs), str(headway),str(virstop),str(test_scenario)))

                                                                            if TRPartA==True:
                                                                                aaa=1
                                                                                if test_scenario==TRPartA_test_scenario_list[0]:
                                                                                    with open(individual_result_dir,
                                                                                              'w+',
                                                                                              newline='') as csvfile_indiv_fixed_only:
                                                                                        fieldnames = ["rq_id", "origin","dest", "dp_time","mode","switch_mode","tt (s)","dist (mi)",
                                                                                                      "tt_C (s)","dist_C (mi)","tt_M (s)","dist_M (mi)",
                                                                                                      "tt_F (s)","dist_F (mi)","tt_W (s)","dist_W (mi)","agent_M_tot_wait_time (s)",
                                                                                                      "agent_F_tot_wait_time (s)","out_of_pock_cost ($)",
                                                                                                      "tt_saved_with_micro (s)","gen_cost_saved_with_micro","mob_logsum_increased_with_micro",
                                                                                                      "agent_prob (auto,transit)","agent_ran_num"]
                                                                                        writer_indiv_fixed_only = csv.DictWriter(
                                                                                            csvfile_indiv_fixed_only,
                                                                                            fieldnames=fieldnames)
                                                                                        writer_indiv_fixed_only.writeheader()

                                                                                        for rq_id in travel_time_delta:

                                                                                            mode = off_micro_ag_mode[
                                                                                                rq_id]

                                                                                            origin = agent_O[rq_id]
                                                                                            dest = agent_D[rq_id]
                                                                                            dp_time = agent_rq_time[
                                                                                                rq_id]

                                                                                            tt_M = 0
                                                                                            dist_M = 0
                                                                                            tt_F = 0
                                                                                            dist_F = 0
                                                                                            tt_C = 0
                                                                                            dist_C = 0
                                                                                            if mode == "T":
                                                                                                tt_C = 0
                                                                                                dist_C = 0
                                                                                                if \
                                                                                                off_micro_agent_F_tot_wait_time[
                                                                                                    rq_id] > 0:
                                                                                                    tt_F = \
                                                                                                    off_micro_ag_tot_tt[
                                                                                                        rq_id]
                                                                                                    dist_F = \
                                                                                                    off_micro_ag_tot_dist[
                                                                                                        rq_id]
                                                                                                else:
                                                                                                    tt_F = 0
                                                                                                    dist_F = 0
                                                                                            else:
                                                                                                tt_C = \
                                                                                                off_micro_ag_tot_tt[
                                                                                                    rq_id]
                                                                                                dist_C = \
                                                                                                off_micro_ag_tot_dist[
                                                                                                    rq_id]
                                                                                                tt_F = 0
                                                                                                dist_F = 0

                                                                                            writer_indiv_fixed_only.writerow(
                                                                                                {"rq_id": rq_id,
                                                                                                 "origin": origin,
                                                                                                 "dest": dest,
                                                                                                 "dp_time": dp_time,
                                                                                                 "mode": mode,
                                                                                                 "switch_mode": 0,
                                                                                                 "tt (s)":
                                                                                                     off_micro_ag_tot_tt[
                                                                                                         rq_id],
                                                                                                 "dist (mi)":
                                                                                                     off_micro_ag_tot_dist[
                                                                                                         rq_id],
                                                                                                 "tt_C (s)": tt_C,
                                                                                                 "dist_C (mi)": dist_C,
                                                                                                 "tt_M (s)": tt_M,
                                                                                                 "dist_M (mi)": dist_M,
                                                                                                 "tt_F (s)": tt_F,
                                                                                                 "dist_F (mi)": dist_F,
                                                                                                 "out_of_pock_cost ($)":
                                                                                                     off_micro_agent_op_cost[
                                                                                                         rq_id],
                                                                                                 "tt_saved_with_micro (s)": 0,
                                                                                                 "gen_cost_saved_with_micro": 0,
                                                                                                 "mob_logsum_increased_with_micro": 0,
                                                                                                 "agent_prob (auto,transit)":
                                                                                                     off_micro_agent_choice_prob[
                                                                                                         rq_id],
                                                                                                 "agent_ran_num":
                                                                                                     off_micro_agent_ran_num[
                                                                                                         rq_id],
                                                                                                 "agent_M_tot_wait_time (s)": 0,
                                                                                                 "agent_F_tot_wait_time (s)":
                                                                                                     off_micro_agent_F_tot_wait_time[
                                                                                                         rq_id],
                                                                                                 "tt_W (s)":
                                                                                                     off_micro_agent_tot_walk_time[
                                                                                                         rq_id],
                                                                                                 "dist_W (mi)":
                                                                                                     off_micro_ag_VMT_W[
                                                                                                         rq_id]})

                                                                                    microtransit_run = microtransit_run_pre

                                                                                    # if TRPartA!=True:
                                                                                    pure_F_users = (
                                                                                                off_micro_T_trips - off_micro_W_trips)

                                                                                    # Write the fixed only scenario
                                                                                    if ((num_virstop_scen == 1) and (num_fleet_size_scen == 1) and (num_operating_periods_scen == 1) and (iteration == 0) and (TRPartA == False)) or ((TRPartA == True) and (test_scenario ==TRPartA_test_scenario_list[0])):
                                                                                        writer.writerow({"test_scen": 0,"dscrptn": "FRT_only_scen", "study_area": study_area,"repositioning": str(False),
                                                                                                         "microtrasnit": "non_micro","hdwy (min)": headway,
                                                                                                         "vir_stop (%)": 0,"flt_sz": 0, "op_periods": 0,"tt_agents": tt_num_agents,
                                                                                                         "car_users": off_micro_auto_trips,"car_mode_share (%)": off_micro_auto_trips / tt_num_agents,
                                                                                                         "trsit_mode_users (W_M_F)": off_micro_T_trips,"transit_mode_share (%)": off_micro_T_trips / tt_num_agents,
                                                                                                         "pure_M_users": 0,"M_trips": 0,"M_mode_share (%)": 0 / tt_num_agents,"pure_F_users": pure_F_users,
                                                                                                         "F_trips": off_micro_F_trips,"F_mode_share (%)": off_micro_F_trips / tt_num_agents,
                                                                                                         "pure_walk_users": off_micro_W_trips,"W_mode_share (%)": off_micro_W_trips / tt_num_agents,
                                                                                                         "walk_users": off_micro_w_trip_legs,"M_pls_F_users": 0,
                                                                                                         "M_pls_F_mode_share (%)": 0,"F_oper_cost ($)": F_oper_cost,
                                                                                                         "M_oper_cost ($)": 0,"F_revenue ($)": off_micro_F_revenue,
                                                                                                         "M_revenue ($)": 0,"Total_T_revenue ($)": off_micro_F_revenue,
                                                                                                         "tot_sub ($)": off_micro_tt_sub,"sub_per_F_trip ($)": off_micro_F_sub_per_ridership,
                                                                                                         "sub_per_F_rider ($)": off_micro_tt_sub / (pure_F_users),
                                                                                                         "sub_per_M_trip ($)": 0,"sub_per_M_rider ($)": 0,"sub_per_T_trip ($)": off_micro_F_sub_per_ridership,
                                                                                                         "sub_per_T_rider ($)": off_micro_tt_sub / (pure_F_users),
                                                                                                         "sub_per_M_pax_mile ($/mi)": 0,"sub_per_F_pax_mile ($/mi)": off_micro_tt_sub / off_micro_VMT_F,
                                                                                                         "sub_per_T_pax_mile ($/mi)": off_micro_tt_sub / off_micro_VMT_F,"sub_per_M_VMT ($/mi)": 0,
                                                                                                         "sub_per_F_VMT ($/mi)": (off_micro_tt_sub) / (transit_tt_VMT),
                                                                                                         "sub_per_T_VMT ($/mi)": (off_micro_tt_sub) / (transit_tt_VMT),
                                                                                                         "tt_auto_gas_cost ($)": off_micro_total_op_cost - off_micro_F_revenue,
                                                                                                         "auto_gas_cost_per_mile ($/mi)": (off_micro_total_op_cost - off_micro_F_revenue) / off_micro_VMT_auto,
                                                                                                         "avg_M_fare": 0,"avg_F_fare": off_micro_F_revenue / (pure_F_users),
                                                                                                         "avg_T_fare": off_micro_F_revenue / (pure_F_users),
                                                                                                         "avg_auto_gas_cost": (off_micro_total_op_cost - off_micro_F_revenue) / off_micro_auto_trips,
                                                                                                         "tt_o_pckt_cost ($)": off_micro_total_op_cost,"tt_mob_lgsm_inc_with_micro": 0,
                                                                                                         "tt_gen_cost": off_micro_tt_gen_cost,"tt_mode_switch": 0,
                                                                                                         "M_avg_wait_time (s)": 0,"F_avg_wait_time (s)": off_micro_avg_F_wait_time,
                                                                                                         "avg_walk_time (s)": off_micro_avg_walk_time,"tt_walk_time (h)": off_micro_tot_walk_time / 3600,
                                                                                                         "car_VMT (mi)": off_micro_VMT_auto,"M_VMT (mi)": 0,
                                                                                                         "M_PMT (mi)": 0,"M_PMT/M_VMT": 0,"F_VMT (mi)": transit_tt_VMT,"F_PMT (mi)": off_micro_VMT_F,"F_PMT/F_VMT": off_micro_VMT_F / transit_tt_VMT,
                                                                                                         "tt_VMT (mi)": off_micro_tt_VMT,"tt_walk_dist (mi)": off_micro_VMT_W,
                                                                                                         "wghted_acc_emp_5_min": tot_weighted_5_min_F,"wghted_acc_emp_10_min": tot_weighted_10_min_F,
                                                                                                         "wghted_acc_emp_15_min": tot_weighted_15_min_F,"M_util_rate (%)": 0,
                                                                                                         "M_veh_occ": 0,"M_avg_speed (mph)": 0,"cnvrg (iter, sm_sq_per_diff)": 0})
                                                                                    # tot_weighted_5_min_F,tot_weighted_10_min_F


                                                                                else:
                                                                                    aaa=1
                                                                            else:
                                                                                with open(individual_result_dir, 'w+',newline='') as csvfile_indiv_fixed_only:
                                                                                    fieldnames = ["rq_id", "origin", "dest","dp_time", "mode", "switch_mode",
                                                                                                  "tt (s)", "dist (mi)", "tt_C (s)","dist_C (mi)", "tt_M (s)",
                                                                                                  "dist_M (mi)", "tt_F (s)","dist_F (mi)", "tt_W (s)",
                                                                                                  "dist_W (mi)","out_of_pock_cost ($)","tt_saved_with_micro (s)",
                                                                                                  "gen_cost_saved_with_micro","mob_logsum_increased_with_micro",
                                                                                                  "agent_prob (auto,transit)","agent_ran_num","agent_M_tot_wait_time (s)","agent_F_tot_wait_time (s)"]
                                                                                    writer_indiv_fixed_only = csv.DictWriter(csvfile_indiv_fixed_only,fieldnames=fieldnames)
                                                                                    writer_indiv_fixed_only.writeheader()

                                                                                    for rq_id in travel_time_delta:

                                                                                        mode = off_micro_ag_mode[rq_id]

                                                                                        origin = agent_O[rq_id]
                                                                                        dest = agent_D[rq_id]
                                                                                        dp_time = agent_rq_time[rq_id]

                                                                                        tt_M = 0
                                                                                        dist_M = 0
                                                                                        tt_F = 0
                                                                                        dist_F = 0
                                                                                        tt_C = 0
                                                                                        dist_C = 0
                                                                                        if mode == "T":
                                                                                            tt_C = 0
                                                                                            dist_C = 0
                                                                                            if off_micro_agent_F_tot_wait_time[rq_id]>0:
                                                                                                tt_F = off_micro_ag_tot_tt[rq_id]
                                                                                                dist_F = off_micro_ag_tot_dist[rq_id]
                                                                                            else:
                                                                                                tt_F = 0
                                                                                                dist_F = 0
                                                                                        else:
                                                                                            tt_C = off_micro_ag_tot_tt[rq_id]
                                                                                            dist_C = off_micro_ag_tot_dist[rq_id]
                                                                                            tt_F = 0
                                                                                            dist_F = 0


                                                                                        writer_indiv_fixed_only.writerow(
                                                                                            {"rq_id": rq_id, "origin": origin,"dest": dest, "dp_time": dp_time,
                                                                                             "mode": mode,"switch_mode": 0,"tt (s)": off_micro_ag_tot_tt[rq_id],
                                                                                             "dist (mi)": off_micro_ag_tot_dist[rq_id],"tt_C (s)": tt_C,
                                                                                             "dist_C (mi)": dist_C,"tt_M (s)": tt_M,
                                                                                             "dist_M (mi)":dist_M,"tt_F (s)": tt_F,
                                                                                             "dist_F (mi)": dist_F,"out_of_pock_cost ($)": off_micro_agent_op_cost[rq_id],
                                                                                             "tt_saved_with_micro (s)": 0,"gen_cost_saved_with_micro": 0,
                                                                                             "mob_logsum_increased_with_micro":0,"agent_prob (auto,transit)": off_micro_agent_choice_prob[rq_id],
                                                                                             "agent_ran_num": off_micro_agent_ran_num[rq_id],"agent_M_tot_wait_time (s)": 0,"agent_F_tot_wait_time (s)": off_micro_agent_F_tot_wait_time[rq_id],
                                                                                             "tt_W (s)": off_micro_agent_tot_walk_time[rq_id],"dist_W (mi)": off_micro_ag_VMT_W[rq_id]})

                                                                            microtransit_run=microtransit_run_pre

                                                                            # if TRPartA!=True:
                                                                            pure_F_users=(off_micro_T_trips-off_micro_W_trips)

                                                                            # Write the fixed only scenario
                                                                            if ((num_virstop_scen == 1) and (num_fleet_size_scen == 1) and (num_operating_periods_scen == 1) and (iteration == 0) and (TRPartA == False)) or ((TRPartA == True) and (test_scenario ==TRPartA_test_scenario_list[0])):

                                                                                writer.writerow({"test_scen":0,"dscrptn":"FRT_only_scen","study_area":study_area,"repositioning":str(False),"microtrasnit":"non_micro","hdwy (min)": headway, "vir_stop (%)": 0, "flt_sz": 0,"op_periods": 0,
                                                                                             "tt_agents": tt_num_agents,"car_users": off_micro_auto_trips,"car_mode_share (%)": off_micro_auto_trips/tt_num_agents,
                                                                                             "trsit_mode_users (W_M_F)": off_micro_T_trips,"transit_mode_share (%)":off_micro_T_trips/tt_num_agents,"pure_M_users":0, "M_trips": 0,"M_mode_share (%)":0/tt_num_agents,
                                                                                             "pure_F_users":pure_F_users,"F_trips": off_micro_F_trips,"F_mode_share (%)":off_micro_F_trips/tt_num_agents,"pure_walk_users": off_micro_W_trips,"W_mode_share (%)":off_micro_W_trips/tt_num_agents,
                                                                                             "walk_users":off_micro_w_trip_legs, "M_pls_F_users":0,"M_pls_F_mode_share (%)":0,"F_oper_cost ($)": F_oper_cost,
                                                                                             "M_oper_cost ($)": 0,"F_revenue ($)": off_micro_F_revenue, "M_revenue ($)": 0,"Total_T_revenue ($)":off_micro_F_revenue, "tot_sub ($)":off_micro_tt_sub,
                                                                                             "sub_per_F_trip ($)": off_micro_F_sub_per_ridership,"sub_per_F_rider ($)":off_micro_tt_sub/(pure_F_users),"sub_per_M_trip ($)": 0,"sub_per_M_rider ($)":0,"sub_per_T_trip ($)":off_micro_F_sub_per_ridership,"sub_per_T_rider ($)":off_micro_tt_sub/(pure_F_users),
                                                                                             "sub_per_M_pax_mile ($/mi)":0,"sub_per_F_pax_mile ($/mi)":off_micro_tt_sub/off_micro_VMT_F,"sub_per_T_pax_mile ($/mi)":off_micro_tt_sub/off_micro_VMT_F,
                                                                                             "sub_per_M_VMT ($/mi)": 0,"sub_per_F_VMT ($/mi)": (off_micro_tt_sub) / (transit_tt_VMT),"sub_per_T_VMT ($/mi)": (off_micro_tt_sub) / (transit_tt_VMT),
                                                                                             "tt_auto_gas_cost ($)":off_micro_total_op_cost-off_micro_F_revenue,"auto_gas_cost_per_mile ($/mi)":(off_micro_total_op_cost-off_micro_F_revenue)/off_micro_VMT_auto,
                                                                                             "avg_M_fare": 0, "avg_F_fare":off_micro_F_revenue/(pure_F_users),"avg_T_fare":off_micro_F_revenue/(pure_F_users), "avg_auto_gas_cost":(off_micro_total_op_cost-off_micro_F_revenue)/off_micro_auto_trips,
                                                                                            "tt_o_pckt_cost ($)": off_micro_total_op_cost,
                                                                                             "tt_mob_lgsm_inc_with_micro": 0,"tt_gen_cost":off_micro_tt_gen_cost,
                                                                                             "tt_mode_switch": 0,"M_avg_wait_time (s)": 0,
                                                                                             "F_avg_wait_time (s)": off_micro_avg_F_wait_time,"avg_walk_time (s)": off_micro_avg_walk_time,"tt_walk_time (h)":off_micro_tot_walk_time/3600,
                                                                                             "car_VMT (mi)":off_micro_VMT_auto,"M_VMT (mi)":0,"M_PMT (mi)":0,"M_PMT/M_VMT":0,"F_VMT (mi)":transit_tt_VMT,"F_PMT (mi)":off_micro_VMT_F,"F_PMT/F_VMT":off_micro_VMT_F/transit_tt_VMT,"tt_VMT (mi)":off_micro_tt_VMT,"tt_walk_dist (mi)":off_micro_VMT_W,
                                                                                             "wghted_acc_emp_5_min":tot_weighted_5_min_F,
                                                                                             "wghted_acc_emp_10_min":tot_weighted_10_min_F,
                                                                                             "wghted_acc_emp_15_min":tot_weighted_15_min_F,
                                                                                             "M_util_rate (%)": 0, "M_veh_occ": 0,"M_avg_speed (mph)": 0 ,"cnvrg (iter, sm_sq_per_diff)":0})
                                                                            #tot_weighted_5_min_F,tot_weighted_10_min_F

                                                                        #calculate the turn-on micro transit scenario

                                                                        util_rate, veh_occ, total_vmt, empty_vkm, avg_speed_mph = output_metrics.get_fleetpy_eval_output_metrics(repositioning)
                                                                        tt_VMT = VMT_auto + total_vmt + transit_tt_VMT

                                                                        M_gas_cost = total_vmt * M_gas_per_mile
                                                                        M_mode_operating_cost += M_gas_cost

                                                                        M_sub_per_ridership = (M_mode_operating_cost - M_revenue) / num_M_trips # microtransit subsidy per microtransit trips
                                                                        tt_sub = (F_oper_cost + M_mode_operating_cost - F_revenue - M_revenue)
                                                                        tt_sub_per_ridership = tt_sub / (num_F_trips + num_M_trips)

                                                                        if microtransit_run=="micro_only":
                                                                            PMT_VMT_ratio_F = 0
                                                                            headway_pre=headway
                                                                            headway = 0
                                                                            off_micro_tot_walk_time = 0
                                                                        else:
                                                                            PMT_VMT_ratio_F = VMT_F / transit_tt_VMT


                                                                        if test_scenario==0:
                                                                            dscrptn="Base_scen (M+F)"
                                                                        if test_scenario==1:
                                                                            dscrptn="Low-income discount"
                                                                        if test_scenario == 2:
                                                                            dscrptn = "Time-based fare"
                                                                        if test_scenario == 3:
                                                                            dscrptn = "FRT accessibility-based fare"
                                                                        if test_scenario == 4:
                                                                            dscrptn = "Intermodal transfer discount"
                                                                        if test_scenario == 5:
                                                                            dscrptn = "Micro_flat_fare"
                                                                        if test_scenario == 6:
                                                                            dscrptn = "Micro_start+dist_based_fr"
                                                                        if test_scenario == 7:
                                                                            dscrptn = "Intermodal trfr disct+Micro_start+dist_based_fr"
                                                                        if test_scenario == 8:
                                                                            dscrptn = "Time-based fare+no_peak_surchge"
                                                                        if test_scenario == 9:
                                                                            dscrptn = "FRT accessibility-based fare+no_high_FRT_acc_surchge"
                                                                        if test_scenario == 10:
                                                                            dscrptn = "Ritun BO (time-based+intrmdl_trfr+Micro_start+dist_based_fr)"

                                                                        if (pure_F_user + M_F_user)!=0:
                                                                            sub_per_F_rider = (F_oper_cost-F_revenue)/(pure_F_user+M_F_user)
                                                                        else:
                                                                            sub_per_F_rider = 999

                                                                        if (pure_M_user+M_F_user) != 0:
                                                                            sub_per_M_rider=(M_mode_operating_cost-M_revenue)/(pure_M_user+M_F_user)
                                                                        else:
                                                                            sub_per_M_rider = 999

                                                                        if (pure_M_user+M_F_user) != 0:
                                                                            avg_M_fare = M_revenue/(pure_M_user+M_F_user)
                                                                        else:
                                                                            avg_M_fare = 0

                                                                        if (pure_F_user+M_F_user) != 0:
                                                                            avg_F_fare = F_revenue/(pure_F_user+M_F_user)
                                                                        else:
                                                                            avg_F_fare = 0

                                                                        if (pure_M_user + pure_F_user + M_F_user) != 0:
                                                                            avg_T_fare = (M_revenue + F_revenue) / (pure_M_user + pure_F_user + M_F_user)
                                                                        else:
                                                                            avg_T_fare = 0

                                                                        if (VMT_F)!=0:
                                                                            sub_per_F_pax_mile =(F_oper_cost-F_revenue)/(VMT_F)
                                                                        else:
                                                                            sub_per_F_pax_mile = 999
                                                                        writer.writerow({"test_scen":test_scenario,"dscrptn":dscrptn,"study_area":study_area,"repositioning":repositioning,"microtrasnit":microtransit_run,"hdwy (min)": headway, "vir_stop (%)": virstop, "flt_sz": fleet_size, "op_periods": operating_periods,
                                                                             "tt_agents": tt_num_agents,"car_users": num_car_trips,  "car_mode_share (%)": num_car_trips/tt_num_agents,
                                                                             "trsit_mode_users (W_M_F)": num_transit_users ,"transit_mode_share (%)":num_transit_users/tt_num_agents,"pure_M_users":pure_M_user,"M_trips":num_M_trips,"M_mode_share (%)":pure_M_user/tt_num_agents,
                                                                              "pure_F_users":pure_F_user,"F_trips":num_F_trips,"F_mode_share (%)":pure_F_user/tt_num_agents,"pure_walk_users":num_W_trips,"W_mode_share (%)":num_W_trips/tt_num_agents,
                                                                              "walk_users":num_W,"M_pls_F_users":M_F_user,"M_pls_F_mode_share (%)":M_F_user/tt_num_agents, "F_oper_cost ($)": F_oper_cost,
                                                                             "M_oper_cost ($)": M_mode_operating_cost,"F_revenue ($)":F_revenue,"M_revenue ($)":M_revenue,"Total_T_revenue ($)":F_revenue+M_revenue,"tot_sub ($)":tt_sub,
                                                                              "sub_per_F_trip ($)":F_sub_per_ridership,"sub_per_F_rider ($)":sub_per_F_rider, "sub_per_M_trip ($)":M_sub_per_ridership,"sub_per_M_rider ($)":sub_per_M_rider,"sub_per_T_trip ($)":tt_sub_per_ridership,
                                                                              "sub_per_T_rider ($)":tt_sub/(pure_F_user+pure_M_user+M_F_user),"sub_per_M_pax_mile ($/mi)":(M_mode_operating_cost-M_revenue)/(VMT_M),"sub_per_F_pax_mile ($/mi)":sub_per_F_pax_mile,"sub_per_T_pax_mile ($/mi)":(tt_sub)/(VMT_M+VMT_F),
                                                                              "sub_per_M_VMT ($/mi)": (M_mode_operating_cost - M_revenue) / (total_vmt),"sub_per_F_VMT ($/mi)": (F_oper_cost - F_revenue)/(transit_tt_VMT),"sub_per_T_VMT ($/mi)": (tt_sub) / (total_vmt + transit_tt_VMT),
                                                                              "tt_auto_gas_cost ($)":total_op_cost-(F_revenue+M_revenue),"auto_gas_cost_per_mile ($/mi)":(total_op_cost-(F_revenue+M_revenue))/VMT_auto,
                                                                               "avg_M_fare": avg_M_fare, "avg_F_fare":avg_F_fare,"avg_T_fare":avg_T_fare, "avg_auto_gas_cost":(total_op_cost-(F_revenue+M_revenue))/num_car_trips,
                                                                              "tt_o_pckt_cost ($)": total_op_cost,"tt_mob_lgsm_inc_with_micro": total_mob_increase,"tt_gen_cost":tt_gen_cost,
                                                                              "tt_mode_switch": total_switch_mode, "M_avg_wait_time (s)": avg_M_wait_time,
                                                                              "F_avg_wait_time (s)": avg_F_wait_time, "avg_walk_time (s)":avg_walk_time,"tt_walk_time (h)":total_walk_time/3600,
                                                                              "car_VMT (mi)":VMT_auto,"M_VMT (mi)":total_vmt,"M_PMT (mi)":VMT_M,"M_PMT/M_VMT":VMT_M/total_vmt,"F_VMT (mi)":transit_tt_VMT,"F_PMT (mi)":VMT_F,"F_PMT/F_VMT":PMT_VMT_ratio_F,"tt_VMT (mi)":tt_VMT,"tt_walk_dist (mi)":VMT_W,
                                                                              "wghted_acc_emp_5_min":avg_period_total_weighted_5_min,
                                                                                         # "inc_wghted_acc_emp_5_min":avg_period_total_weighted_5_min-tot_weighted_5_min_F,"inc_wghted_acc_emp_5_min_per_sub":(avg_period_total_weighted_5_min-tot_weighted_5_min_F)/tt_sub,
                                                                              "wghted_acc_emp_10_min":avg_period_total_weighted_10_min,
                                                                                         # "inc_wghted_acc_emp_10_min":avg_period_total_weighted_10_min-tot_weighted_10_min_F,"inc_wghted_acc_emp_10_min_per_sub":(avg_period_total_weighted_10_min-tot_weighted_10_min_F)/tt_sub,
                                                                              "wghted_acc_emp_15_min":avg_period_total_weighted_15_min,
                                                                                         # "inc_wghted_acc_emp_15_min":avg_period_total_weighted_15_min-tot_weighted_15_min_F,"inc_wghted_acc_emp_15_min_per_sub":(avg_period_total_weighted_15_min-tot_weighted_15_min_F)/tt_sub,
                                                                                "M_util_rate (%)":util_rate,"M_veh_occ":veh_occ,"M_avg_speed (mph)":avg_speed_mph,"cnvrg (iter, sm_sq_per_diff)":iter_sum_sq_per_diff})


                                                                        if microtransit_run == "micro_only":
                                                                            headway = headway_pre



                                                                        scenario_count += 1
                                                            if converged == False:
                                                                print("running FleetPy....according to demand component result")
                                                                run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
                                                                list_results = read_outputs_for_comparison(cc, sc)
                                                                all_scenario_assert_dict = {0: {"number users": 88}}
                                                                check_assertions(list_results, all_scenario_assert_dict)
                                                                print("FleetPy finishes. Now update the microtransit part in super-network.....","iteration:",iteration)
                                                                update.update_network_files(repositioning,microtransit_run,headway,virstop,M_operating_hrs,fleet_size,study_area,dt_sd_full_trnst_ntwk,zonal_partition,iteration,
                                                                                            iteration_debug,aggregation,TRPartA,BayesianOptimization)


                                                            iteration+=1



                                                        timeatfinished = datetime.datetime.now()
                                                        processingtime = timeatfinished - pre_timeatfinished
                                                        pre_timeatfinished = timeatfinished
                                                        print(scenario_count, "th scenario finished","...virtual_stop:", virstop, "(%)","headway", headway, "(mins)", "fleet_size", fleet_size, "(veh)", "operating_periods",operating_periods,"....... end time is:",timeatfinished,"run time is: ",processingtime)



                csvfile.close()

                timeatfinished = datetime.datetime.now()
                processingtime = timeatfinished - start_time
                print("Study_area",study_area,"integrated system model run finished....... end time is:", timeatfinished, "total run time is: ", processingtime)



        # df_final.to_csv(eval_output_file)
        # # Base Examples with Optimization (requires gurobi license!)
        # # ----------------------------------------------------------
        # b) Pooling in BatchOffer environment
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_pool.csv")
        # # sc = os.path.join(scs_path, "example_pool.csv")
        # # sc = os.path.join(scs_path, "example_pool_1000_rq_5_veh.csv")
        # # sc = os.path.join(scs_path, "example_pool_1000_rq_10_veh.csv")
        # sc = os.path.join(scs_path, "example_pool_1000_rq_20_veh.csv")
        # # sc = os.path.join(scs_path, "example_pool_1000_rq_30_veh.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        # list_results = read_outputs_for_comparison(cc, sc)
        # all_scenario_assert_dict = {0: {"number users": 91}}
        # check_assertions(list_results, all_scenario_assert_dict)

        # # c) Pooling in ImmediateOfferEnvironment
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_ir.csv")
        # sc = os.path.join(scs_path, "example_ir_batch.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        # list_results = read_outputs_for_comparison(cc, sc)
        # all_scenario_assert_dict = {0: {"number users": 90}}
        # check_assertions(list_results, all_scenario_assert_dict)
        #
        # # d) Pooling with RV heuristics in ImmediateOfferEnvironment (with doubled demand)
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_ir.csv")
        # t0 = time.perf_counter()
        # # no heuristic scenario
        # sc = os.path.join(scs_path, "example_pool_noheuristics.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        # list_results = read_outputs_for_comparison(cc, sc)
        # all_scenario_assert_dict = {0: {"number users": 199}}
        # check_assertions(list_results, all_scenario_assert_dict)
        # # with heuristic scenarios
        # t1 = time.perf_counter()
        # sc = os.path.join(scs_path, "example_pool_heuristics.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        # list_results = read_outputs_for_comparison(cc, sc)
        # t2 = time.perf_counter()
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=2)
        # t3 = time.perf_counter()
        # print(f"Computation time without heuristics: {round(t1-t0, 1)} | with heuristics 1 CPU: {round(t2-t1,1)}"
        #       f"| with heuristics 2 CPU: {round(t3-t2,1)}")
        # all_scenario_assert_dict = {0: {"number users": 191}}
        # check_assertions(list_results, all_scenario_assert_dict)
        #
        # g) Pooling with RV heuristic and Repositioning in ImmediateOfferEnvironment (with doubled demand and
        #       bad initial vehicle distribution)
    else:
        log_level = "info"
        scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_study", "scenarios")
        cc = os.path.join(scs_path, "constant_config_ir_repo.csv")
        sc = os.path.join(scs_path, "example_ir_heuristics_repositioning.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 198}}
        check_assertions(list_results, all_scenario_assert_dict)
        #
        # # h) Pooling with public charging infrastructure (low range vehicles)
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_charge.csv")
        # sc = os.path.join(scs_path, "example_charge.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        #
        # # i) Pooling and active vehicle fleet size is controlled externally (time and utilization based)
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_depot.csv")
        # sc = os.path.join(scs_path, "example_depot.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        #
        # # j) Pooling with public charging and fleet size control (low range vehicles)
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_depot_charge.csv")
        # sc = os.path.join(scs_path, "example_depot_charge.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        #
        # # h) Pooling with multiprocessing
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_depot_charge.csv")
        # # no heuristic scenario single core
        # t0 = time.perf_counter()
        # sc = os.path.join(scs_path, "example_depot_charge.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        # list_results = read_outputs_for_comparison(cc, sc)
        # all_scenario_assert_dict = {0: {"number users": 199}}
        # check_assertions(list_results, all_scenario_assert_dict)
        # print("Computation without multiprocessing took {}s".format(time.perf_counter() - t0))
        # # no heuristic scenario multiple cores
        # cores = 2
        # t0 = time.perf_counter()
        # sc = os.path.join(scs_path, "example_depot_charge.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=cores, n_parallel_sim=1)
        # list_results = read_outputs_for_comparison(cc, sc)
        # all_scenario_assert_dict = {0: {"number users": 199}}
        # check_assertions(list_results, all_scenario_assert_dict)
        # print("Computation with multiprocessing took {}s".format(time.perf_counter() - t0))
        # print(" -> multiprocessing only usefull for large vehicle fleets")
        #
        # # j) Pooling - multiple operators and broker
        # log_level = "info"
        # cc = os.path.join(scs_path, "constant_config_broker.csv")
        # sc = os.path.join(scs_path, "example_broker.csv")
        # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        #
