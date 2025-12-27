import sys
sys.path.append('recovery/SGATGANSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .SGATGANSrc.src.constants import *
from .SGATGANSrc.src.utils import *
from .SGATGANSrc.src.train import *

# import sys
# import os
# # 将 SGATGANSrc 的父目录添加到模块搜索路径
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'SGATGANSrc')))
# from SGATGANSrc.src.train import class_accuracy, accuracy
from .SGATGANSrc.src.train import class_accuracy, accuracy
class SGATGANRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.model_name = f'FEE_{hosts}'
        self.gen_name = f'Gen_{hosts}'
        self.disc_name = f'Disc_{hosts}'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.save_gan = True   ############
        self.load_models()

    def load_models(self):
        # Load encoder model

        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        print('loadddddddddddddddddddddddddddddddddd', self.epoch,self.model)
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model)
        # Load generator and discriminator
        self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
            load_gan(model_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', self.gen_name, self.disc_name)
        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        # Freeze GAN if not training
        # if not self.training: freeze(self.gen); freeze(self.disc)
        # if self.training:  self.ganloss = nn.BCELoss()
        self.ganloss = nn.BCELoss()   ###############
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    ######### 一个类别准确率
    def train_model(self):

        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        print('sdasdasdasdasd', self.env_name, self.model_name,folder)
        train_time_data, train_schedule_data, anomaly_data, class_data, thresholds = load_dataset(folder, self.model)
        print('789789789',class_data)
        # train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 构建 result.txt 的路径
        result_file = os.path.join(project_root, "result.txt")
        # 用于记录结果的文件路径
        print('trainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn;nnnnnnnnnnnnnnn',result_file)
        best_anomaly_score, best_class_score = 0, 0
        best_anomaly_epoch, best_class_epoch = 0, 0
        best_precision, best_recall, best_f1_score = 0, 0, 0  # 新增最佳指标变量
        best_precision_epoch, best_recall_epoch, best_f1_epoch = 0, 0, 0  # 新增最佳指标对应的epoch

        # 打开结果文件写入结果
        with open(result_file, "a") as f:
            f.write("Epoch\tAnomaly_Accuracy\tClass_Accuracy\tPrecision\tRecall\tF1-Score\n")

            for self.epoch in tqdm(range(self.epoch + 1, self.epoch + num_epochs + 1), position=0):
                # 训练步骤
                loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data,
                                        class_data, self.optimizer)

                # 计算精度及指标
                anomaly_score, class_score, precision, recall, f1_score = accuracy(self.model, train_time_data,
                                                                                   train_schedule_data,
                                                                                   anomaly_data, class_data,
                                                                                   self.model_plotter)

                # 更新最佳批次记录
                if anomaly_score > best_anomaly_score:
                    best_anomaly_score = anomaly_score
                    best_anomaly_epoch = self.epoch

                if class_score > best_class_score:
                    best_class_score = class_score
                    best_class_epoch = self.epoch

                # 更新最佳精度、召回率和F1分数
                if precision > best_precision:
                    best_precision = precision
                    best_precision_epoch = self.epoch

                if recall > best_recall:
                    best_recall = recall
                    best_recall_epoch = self.epoch

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_f1_epoch = self.epoch

                # 写入结果文件
                f.write(f"{self.epoch}\t{anomaly_score:.4f}\t{class_score:.4f}\t"
                        f"{precision:.4f}\t{recall:.4f}\t{f1_score:.4f}\n")

                tqdm.write(
                    f"Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}")
                self.accuracy_list.append((loss, factor, anomaly_score, class_score))
                self.model_plotter.plot(self.accuracy_list, self.epoch)

                # 保存模型
                save_model(model_folder, f"{self.env_name}_{self.model_name}.ckpt", self.model, self.optimizer,
                           self.epoch, self.accuracy_list)

            # 写入最佳批次信息
            f.write("\nBest Anomaly Detection:\n")
            f.write(f"Epoch: {best_anomaly_epoch}, Accuracy: {best_anomaly_score:.4f}\n")
            f.write("Best Classification:\n")
            f.write(f"Epoch: {best_class_epoch}, Accuracy: {best_class_score:.4f}\n")
            f.write("\nBest Precision:\n")
            f.write(f"Epoch: {best_precision_epoch}, Precision: {best_precision:.4f}\n")
            f.write("Best Recall:\n")
            f.write(f"Epoch: {best_recall_epoch}, Recall: {best_recall:.4f}\n")
            f.write("Best F1 Score:\n")
            f.write(f"Epoch: {best_f1_epoch}, F1 Score: {best_f1_score:.4f}\n")


    def train_gan(self, embedding, schedule_data):
        # Train discriminator
        self.disc.zero_grad()
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data.detach())
        new_score, orig_score = run_simulation(self.env.stats, new_schedule_data), run_simulation(self.env.stats, schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) if new_score <= orig_score else torch.tensor([1, 0], dtype=torch.double)
        disc_loss = self.ganloss(probs, true_probs.detach().clone())
        disc_loss.backward(); self.dopt.step()
        # Train generator
        self.gen.zero_grad()
        probs = self.disc(schedule_data, new_schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) # to enforce new schedule is better than original schedule
        gen_loss = self.ganloss(probs, true_probs)
        gen_loss.backward(); self.gopt.step()
        # Append to accuracy list
        if self.save_gan:    ####
            self.epoch += 1; self.accuracy_list.append((gen_loss.item(), disc_loss.item()))
            print(f'{color.HEADER}Epoch1 {self.epoch},\tGLoss1 = {gen_loss.item()},\tDLoss1 = {disc_loss.item()}{color.ENDC}')
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)
            save_gan(model_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', \
                    self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data)
        self.gan_plotter.new_better(probs[1] >= probs[0])
        if probs[0] > probs[1]: # original better
            return original_decision
        # Form new decision
        host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)): host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1:
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision); hosts_from = [0] * self.hosts
        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            one_hot = schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            if container_alloc[cid] != new_host:
                decision_dict[cid] = new_host
                hosts_from[container_alloc[cid]] = 1
        self.gan_plotter.plot_test(hosts_from)
        return list(decision_dict.items())

    def run_encoder(self, schedule_data):
        # Get latest data from Stat
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return self.model(time_data, schedule_data)

    def run_model(self, time_series, original_decision):
        # Run encoder
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        anomaly, prototype = self.run_encoder(schedule_data)
        # If no anomaly predicted, return original decision
        for a in anomaly:
            prediction = torch.argmax(a).item()
            if prediction == 1:
                self.gan_plotter.update_anomaly_detected(1)
                break
        else:
            self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        # Pass through GAN
        #if self.training:
        self.train_gan(embedding, schedule_data)
        #self.tune_model()
            # return original_decision
        return self.recover_decision(embedding, schedule_data, original_decision)