import os
import time
import json
import shutil
import random
import pickle
import traceback
import pickle as pkl
from copy import deepcopy
from multiprocessing import Process

from updater import Updater
from generator import Generator
from construct_sample import ConstructSample


def goodness_fairshare(state_dist, num_inter=1):
    sm = 0
    for n in range(num_inter):
        density = pkl.load(state_dist)
        for i in range(len(density)):
            for j in range(len(density[0])):
                dist = abs(((j + 1) // 3) - i) + 1
                sm += sum(density[i][j]) / dist
    return sm

def goodness_decisionconsistency(state_dist, num_inter=1):
    sm = 0
    for n in range(num_inter):
        density = pkl.load(state_dist)
        for i in range(len(density)):
            alpha = len(density[0]) // 10
            cnt = [0,0]
            mode = 0
            for j in range(len(density[0])):
                if mode==0:
                    if density[i][j]:
                        cnt[0] = 0
                    else:
                        cnt[0] += 1
                        if cnt[0] > alpha:
                            mode = 1
                elif mode == 1:
                    if density[i][j]==0:
                        cnt[0] += 1
                    else:
                        cnt[1] += 1
                        mode = 2
                else:
                    if density[i][j]:
                        cnt[1] += 1
                    else:
                        break
            sm += cnt[0] / (cnt[1]+1)
    return sm

def goodness(typ, state_dist, num_inter=1):
    if typ == 1:
        return goodness_fairshare(state_dist, num_inter)
    if typ == 2:
        return goodness_decisionconsistency(state_dist, num_inter)
    raise 'Unkown goodness %d' % typ

class Pipeline:

    def _copy_conf_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"), indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
        json.dump(self.dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                    os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["TRAFFIC_FILE"]),
                    os.path.join(path, self.dic_traffic_env_conf["TRAFFIC_FILE"]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        # do file operations
        self._copy_conf_file()
        self._copy_anon_file()
        self.test_duration = []

        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)
        self.no_test = dic_traffic_env_conf['NO_TEST'] if 'NO_TEST' in dic_traffic_env_conf.keys() else 0

        self.dic_agent_conf_test = deepcopy(self.dic_agent_conf)
        self.dic_agent_conf_test["EPSILON"] = 0
        self.dic_agent_conf_test["MIN_EPSILON"] = 0

    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, isTest=False):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              isTest=isTest
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path):
        updater = Updater(cnt_round=cnt_round,
                          dic_agent_conf=dic_agent_conf,
                          dic_exp_conf=dic_exp_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          dic_path=dic_path
                          )
        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")

    def downsample(self, path_to_log, i):
        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                os.remove(path_to_pkl)
                with open(path_to_pkl, "wb") as f_subset:
                    try:
                        pickle.dump(subset_data, f_subset)
                    except Exception as e:
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
            except Exception as e:
                print("Error occurs when READING pickles when down sampling for inter {0}".format(i))
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.downsample(path_to_log, i)

    def construct_sample_multi_process(self, train_round, cnt_round, batch_size=200):
        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        if batch_size > self.dic_traffic_env_conf['NUM_INTERSECTIONS']:
            batch_size_run = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'], batch_size_run):
            start = batch
            stop = min(batch + batch_size, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])
            process_list.append(Process(target=self.construct_sample_batch, args=(cs, start, stop)))

        for t in process_list:
            t.start()
        for t in process_list:
            t.join()

    def construct_sample_batch(self, cs, start,stop):
        for inter_id in range(start, stop):
            print("make construct_sample_wrapper for ", inter_id)
            cs.make_reward(inter_id)

    def process_goodness(self, cnt_round):
        num_inter = self.dic_traffic_env_conf['NUM_AGENTS']
        file_gn = os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "train_round", "goodness.pkl")
        file_gn_txt = self.dic_path['PATH_TO_WORK_DIRECTORY'] + '/train_round/gn.txt'
        if os.path.exists(file_gn):
            gn = pkl.load(open(file_gn, 'rb'))
        else:
            gn = {'GN_IDX': 0,
                 'GN_THETA': 0.0,
                 'GN_IDX1': 0,
                 'GN_THETA1': 0.0}

        gn_state_dist = open(os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "test_round", "round_" + str(cnt_round), "lut_model.pkl" if self.dic_agent_conf['GOODNESS'] == 2 else "state_dist.pkl"), "rb")
        # Load the header
        pkl.load(gn_state_dist)

        keep_best = 1
        train_dir = os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "train_round", "round_" + str(cnt_round))
        test_dir = os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "test_round", "round_" + str(cnt_round))

        gnThetaOld = gn['GN_THETA']
        gnThetaDecay = gnThetaOld * (1 - gn['GN_IDX']*self.dic_agent_conf["DECAY"])
        gnTheta = goodness(self.dic_agent_conf['GOODNESS'], gn_state_dist, num_inter)

        path_bkup = self.dic_path["PATH_TO_WORK_DIRECTORY"] + '/train_round/bkup'
        path_bkup_tst = self.dic_path["PATH_TO_WORK_DIRECTORY"] + '/test_round/bkup'
        if not os.path.exists(path_bkup):
            os.makedirs(path_bkup)
            os.makedirs(path_bkup_tst)
            for i in range(self.dic_agent_conf['ETA']):
                os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"]+'/model/SubRound_'+str(i))

        done = True
        done2 = gn['GN_IDX'] and gn['GN_THETA1'] >= gnThetaDecay
        if gnTheta >= gnThetaDecay:
            done2 = False
        elif not done2:
            os.system('mv {0} {0}_{1}'.format(train_dir, gn['GN_IDX']))
            os.system('mv {0} {0}_{1}'.format(test_dir, gn['GN_IDX']))
            os.system('mv {0}/model/round_{1}_inter_*.h5 {0}/model/SubRound_{2}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], cnt_round, gn['GN_IDX']))
            os.system('cp {0}/*.csv {1}_{2}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], train_dir, gn['GN_IDX']))
            os.system('cp {0}/train_round/total_samples_inter_*.pkl {1}_{2}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], train_dir, gn['GN_IDX']))
            os.system('cp {0}/test_round/total_samples_inter_*.pkl {1}_{2}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], test_dir, gn['GN_IDX']))

            if gnTheta > gn['GN_THETA1']:
                gn['GN_THETA1'] = gnTheta
                gn['GN_IDX1'] = gn['GN_IDX']
            gn['GN_IDX'] += 1

            if gn['GN_IDX'] < self.dic_agent_conf['ETA']:
                done = False
                # Revert metrics and files
                os.system('cp {1}/*.csv {0}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], path_bkup))
                os.system('cp {1}/total_samples_inter_*.pkl {0}/train_round/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"],path_bkup))
                os.system('cp {1}/total_samples_inter_*.pkl {0}/test_round/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"],path_bkup_tst))
                print(cnt_round, gnTheta, file=open(file_gn_txt, 'a'))
            else:
                print(cnt_round, gnTheta, -1, file=open(file_gn_txt, 'a'))
                done2 = True

        if done2:
            idx = gn['GN_IDX1'] if keep_best else self.dic_agent_conf['GN_IDX']-1
            gnTheta = gn['GN_THETA1']
            os.system('ln -s round_{0}_{1} {2}'.format(cnt_round, idx, train_dir))
            os.system('ln -s round_{0}_{1} {2}'.format(cnt_round, idx, test_dir))
            os.system('mv {0}/model/SubRound_{2}/round_{1}_inter_*.h5 {0}/model/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], cnt_round, idx))
            os.system('cp {1}/*.csv {0}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], train_dir))
            os.system('cp {1}_{2}/total_samples_inter_*.pkl {0}/train_round/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], train_dir, idx))
            os.system('cp {1}_{2}/total_samples_inter_*.pkl {0}/test_round/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"], test_dir, idx))

        if done:
            print(cnt_round, gnTheta, gn['GN_IDX1'], file=open(file_gn_txt, 'a'))

            gn['GN_THETA'] = gnTheta
            gn['GN_IDX'] = 0
            gn['GN_THETA1'] = 0.0
            gn['GN_IDX1'] = 0

            print('gn id:{0} theta:{1:.2f}->{2:.2f}->{3:.2f}'.format(id, gnThetaOld, gnThetaDecay, gnTheta))
            os.system('cp {0}/*.csv {1}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"],path_bkup))
            os.system('cp {0}/train_round/total_samples_inter_*.pkl {1}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"],path_bkup))
            os.system('cp {0}/test_round/total_samples_inter_*.pkl {1}/'.format(self.dic_path["PATH_TO_WORK_DIRECTORY"],path_bkup_tst))

        pkl.dump(gn, open(file_gn, 'wb'))

    def generate_samples(self, cnt_round, multi_process, isTest=True, num_gen=1):
        process_list = []
        generator_start_time = time.time()
        dic_agent_conf = self.dic_agent_conf_test if isTest else self.dic_agent_conf

        if multi_process:
            for cnt_gen in range(num_gen):
                p = Process(target=self.generator_wrapper,
                            args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                  dic_agent_conf, self.dic_traffic_env_conf, isTest)
                            )
                print("before")
                p.start()
                print("end")
                process_list.append(p)
            print("before join")
            for i in range(len(process_list)):
                p = process_list[i]
                print("generator %d to join" % i)
                p.join()
                print("generator %d finish join" % i)
            print("end join")
        else:
            for cnt_gen in range(num_gen):
                self.generator_wrapper(cnt_round=cnt_round,
                                       cnt_gen=cnt_gen,
                                       dic_path=self.dic_path,
                                       dic_exp_conf=self.dic_exp_conf,
                                       dic_agent_conf=dic_agent_conf,
                                       dic_traffic_env_conf=self.dic_traffic_env_conf,
                                       isTest=isTest)
        generator_total_time = time.time() - generator_start_time
        print("==============  make samples =============")
        # make samples and determine which samples are good
        making_samples_start_time = time.time()
        round_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round" if isTest else "train_round")
        if not os.path.exists(round_dir):
            os.makedirs(round_dir)
        cs = ConstructSample(path_to_samples=round_dir, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        cs.make_reward_for_system()

        making_samples_total_time = time.time() - making_samples_start_time

        if isTest and self.dic_agent_conf['GOODNESS']:
            self.process_goodness(cnt_round)

        return generator_total_time, making_samples_total_time

    def run(self, multi_process=False):
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        cnt_round = self.dic_exp_conf["START_ROUNDS"]-1
        while cnt_round < self.dic_exp_conf["START_ROUNDS"]+self.dic_exp_conf["NUM_ROUNDS"]-1:
            cnt_round += 1
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                print("==============  generator =============")
                generator_total_time, making_samples_total_time = self.generate_samples(cnt_round, multi_process, False, self.dic_exp_conf["NUM_GENERATORS"])

                print("==============  update network =============")
                update_network_start_time = time.time()
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path
                                      ))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path)

                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                    self.downsample_for_system(path_to_log,self.dic_traffic_env_conf)
                update_network_end_time = time.time()
                update_network_total_time = update_network_end_time - update_network_start_time

            test_evaluation_total_time = 0
            if cnt_round >= self.no_test:
                generator_total_time2, making_samples_total_time2 = self.generate_samples(cnt_round, multi_process)
                test_evaluation_total_time = generator_total_time2 + making_samples_total_time2

            if "generator_total_time" in locals():
                print("Generator time:", generator_total_time)
                print("Making samples time:", making_samples_total_time)
                print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))

            if "generator_total_time" in locals():
                f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")
                f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time,making_samples_total_time,
                                                              update_network_total_time,test_evaluation_total_time,
                                                              time.time()-round_start_time))
                f_time.close()

            if self.dic_agent_conf['GOODNESS'] and not os.path.exists(os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'], "train_round", "round_" + str(cnt_round))):
                cnt_round -= 1
                print('********* ReDoing round', cnt_round, '************')
