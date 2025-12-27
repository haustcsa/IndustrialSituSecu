# from .constants import *
from recovery.SGATGANSrc.src.constants import *
# from .utils import *
from recovery.SGATGANSrc.src.utils import *
import torch.nn as nn
from tqdm import tqdm
# from .plotter import *
from recovery.SGATGANSrc.src.plotter import *
anomaly_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss(reduction = 'mean')

num_zero, num_ones = 1, 1

# Model Training
def triplet_loss(anchor, positive_class, model):
	#print('recovert888888888')
	global PROTO_UPDATE_FACTOR
	positive_loss = mse_loss(anchor, model.prototype[positive_class].detach().clone())
	negative_class_list = [0, 1, 2]
	negative_class_list.remove(positive_class)
	negative_loss = []
	for nc in negative_class_list:
		negative_loss.append(mse_loss(anchor, model.prototype[nc]))
	loss = positive_loss - torch.sum(torch.tensor(negative_loss))
	if positive_loss <= negative_loss[0] and positive_loss <= negative_loss[1]:
		factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
		model.prototype[positive_class] = factor * anchor + (1 - factor) * model.prototype[positive_class]
	return loss

def custom_loss(model, source, target_anomaly, target_class):
	global PROTO_UPDATE_FACTOR, num_ones, num_zero
	nz, no = 0, 0
	source_anomaly, source_prototype = source
	aloss, tloss = 0, torch.tensor(0, dtype=torch.double)
	for i, sa in enumerate(source_anomaly):
		multiplier = 1 if target_anomaly[i] == 0 else num_zero / num_ones
		nz += 1 if target_anomaly[i] == 0 else 1; no += 1 if target_anomaly[i] == 1 else 0
		aloss += anomaly_loss(sa,  torch.tensor([target_anomaly[i]], dtype=torch.long)) * multiplier
	for i, sp in enumerate(source_prototype):
		if target_anomaly[i] > 0:
			tloss += triplet_loss(sp, target_class[i], model)
	PROTO_UPDATE_FACTOR *= PROTO_FACTOR_DECAY; num_zero += nz; num_ones += no;
	return aloss, tloss

def backprop(epoch, model, train_time_data, train_schedule_data, anomaly_data, class_data, optimizer, training = True):
	global PROTO_UPDATE_FACTOR, num_ones, num_zero
	num_zero, num_ones = 1, 1
	aloss_list, tloss_list = [], []
	for i in tqdm(range(train_time_data.shape[0]), leave=False, position=1):
		output = model(train_time_data[i], train_schedule_data[i])
		aloss, tloss = custom_loss(model, output, anomaly_data[i], class_data[i])
		aloss_list.append(aloss.item()); tloss_list.append(tloss.item())
		loss = aloss + tloss
		if training:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(aloss_list)+np.mean(tloss_list)},\tALoss = {np.mean(aloss_list)},\tTLoss = {np.mean(tloss_list)}')
	factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
	return np.mean(aloss_list) + np.mean(tloss_list), factor

# Accuracy 
def anomaly_accuracy(source_anomaly, target_anomaly, model_plotter):
	correct = 0; res_list = []; tp, fp, tn, fn = 0, 0, 0, 0
	for i, sa in enumerate(source_anomaly):
		#print('iiiiiiiiii', i)
		#print('iiiiiiiiii1', sa)
		#print('iiiiiiiiii2222222222', source_anomaly)
		res = torch.argmax(sa).item()
		#print('aaaaaaaa',res,target_anomaly[i])
		res_list.append(res)
		if res == target_anomaly[i]:
			correct += 1
			if target_anomaly[i] == 1: tp += 1
			else: tn += 1
		else:
			if target_anomaly[i] == 1: fn += 1
			else: fp += 1
	if model_plotter is not None:
		model_plotter.update_anomaly(res_list, target_anomaly, correct/len(source_anomaly))
	return correct/len(source_anomaly), tp, tn, fp, fn


########################原
def class_accuracy(source_prototype, target_anomaly, target_class, model, model_plotter):  # 计算分类精度：通过比较模型的原型和目标类别的原型来判断是否分类正确
	correct, total = 0, 1e-4; prototypes = []
	for i, sp in enumerate(source_prototype):
		# print('sadasda', sp)
		if target_anomaly[i] > 0:
			total += 1
			# print('1213',prototypes,model.prototype)
			positive_loss = mse_loss(sp, model.prototype[target_class[i]])
			# print(f"i={i}, positive_loss={positive_loss}")
			# print('2222222222',target_class[i])
			negative_class_list = [0, 1, 2]
			negative_class_list.remove(target_class[i])
			negative_loss = []
			for nc in negative_class_list:
				negative_loss.append(mse_loss(sp, model.prototype[nc]))
			if positive_loss <= negative_loss[0] and positive_loss <= negative_loss[1]:
				correct += 1
			prototypes.append((sp, target_class[i]))
	if model_plotter is not None:
		model_plotter.update_class(prototypes, correct/total)
	return correct / total



###########多类别 牛版
# def class_accuracy(source_prototype, target_anomaly, target_class, model, model_plotter):
#     """
#     计算每个类别的分类精度，分别统计类别 0, 1, 2 的准确率
#     """
#     correct = {0: 0, 1: 0, 2: 0}  # 每个类别正确分类的数量
#     total = {0: 0, 1: 0, 2: 0}  # 每个类别的总样本数
#     prototypes = []
#
#     for i, sp in enumerate(source_prototype):
#         if target_anomaly[i] > 0:  # 仅计算异常样本
#             cls = target_class[i]  # 当前样本的真实类别
#             total[cls] += 1  # 统计当前类别的样本总数
#
#             # 计算 MSE（当前样本与其真实类别的原型之间的距离）
#             positive_loss = mse_loss(sp, model.prototype[cls])
#
#             # 获取负类别
#             negative_class_list = [nc for nc in [0, 1, 2] if nc != cls]
#             negative_loss = [mse_loss(sp, model.prototype[nc]) for nc in negative_class_list]
#
#             # 判断是否分类正确（MSE 最小）
#             if positive_loss <= min(negative_loss):
#                 correct[cls] += 1  # 该类别预测正确
#
#             prototypes.append((sp, cls))  # 记录该样本原型
#
#     # 计算每个类别的准确率，防止除零错误
#     accuracy_per_class = {
#         cls: correct[cls] / total[cls] if total[cls] > 0 else 0
#         for cls in [0, 1, 2]
#     }
#
#     # 如果提供了 `model_plotter`，更新可视化
#     if model_plotter is not None:
#         model_plotter.update_class(prototypes, accuracy_per_class)
#
#     return accuracy_per_class  # 返回每个类别的准确率



########## 多类别  宋小宝版
# def class_accuracy(source_prototype, target_anomaly, target_class, model, model_plotter):
# 	# 初始化每个类别的 TP、FP、FN
# 	num_classes = 3  # 假设有三个类别
# 	tp = [0] * num_classes  # 真阳性
# 	fp = [0] * num_classes  # 假阳性
# 	fn = [0] * num_classes  # 假阴性
# 	total = 1e-4  # 避免除以零
# 	correct = 0
# 	prototypes = []
#
# 	# 遍历样本
# 	for i, sp in enumerate(source_prototype):
# 		if target_anomaly[i] > 0:  # 只处理异常样本
# 			total += 1
# 			true_class = target_class[i]  # 真实类别
# 			positive_loss = mse_loss(sp, model.prototype[true_class])
#
# 			# 计算与所有类别的损失，并分离梯度
# 			all_losses = [mse_loss(sp, model.prototype[c]).detach().item() for c in range(num_classes)]
# 			pred_class = np.argmin(all_losses)  # 预测的类别
#
# 			# 更新混淆矩阵统计量
# 			if pred_class == true_class:
# 				tp[true_class] += 1  # 预测正确
# 				correct += 1
# 			else:
# 				fp[pred_class] += 1  # 假阳性（预测的类别）
# 				fn[true_class] += 1  # 假阴性（真实类别）
#
# 			prototypes.append((sp, true_class))
#
# 	# 计算每个类别的 Recall、Precision 和 F1
# 	recall = []
# 	precision = []
# 	f1 = []
# 	for c in range(num_classes):
# 		# Recall = TP / (TP + FN)
# 		recall_c = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
# 		# Precision = TP / (TP + FP)
# 		precision_c = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
# 		# F1 = 2 * Precision * Recall / (Precision + Recall)
# 		f1_c = (2 * precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0
#
# 		recall.append(recall_c)
# 		precision.append(precision_c)
# 		f1.append(f1_c)
# 	# 更新可视化（如果需要）
# 	if model_plotter is not None:
# 		model_plotter.update_class(prototypes, correct / total)
# 	# 返回整体准确率和每个类别的指标
# 	#return correct / total
# 	return {
# 		'accuracy': correct / total,
# 		'recall': recall,  # 每个类别的 Recall
# 		'precision': precision,  # 每个类别的 Precision
# 		'f1': f1  # 每个类别的 F1
# 	}


#######原
# def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data, model_plotter):
# 	anomaly_correct, class_correct, class_total = 0, 0, 0; tpl, tnl, fpl, fnl = [], [], [], []
# 	for i, d in enumerate(train_time_data):
# 		output = model(train_time_data[i], train_schedule_data[i])
# 		source_anomaly, source_prototype = output
# 		# 计算异常准确度
# 		res, tp, tn, fp, fn = anomaly_accuracy(source_anomaly, anomaly_data[i], model_plotter)
# 		anomaly_correct += res
# 		tpl.append(tp); tnl.append(tn); fpl.append(fp); fnl.append(fn)
# 		tp += res; fp += res; tn += (1 - res); fn += (1 - res)
# 		# 分类准确度
# 		if np.sum(anomaly_data[i]) > 0:
# 			class_total += 1
# 			class_correct += class_accuracy(source_prototype, anomaly_data[i], class_data[i], model, model_plotter)
# 	tp, fp, tn, fn = np.mean(tpl), np.mean(fpl), np.mean(tnl), np.mean(fn)
# 	p, r = tp/(tp+fp), tp/(tp+fn)    # precision = tp / (tp + fp) if (tp + fp) != 0 else 0
# 	f1=2 * p * r / (p + r)
# 	tqdm.write(f'P = {p}, R = {r}, F1 = {f1}')
# 	return anomaly_correct / len(train_time_data), class_correct / class_total,p,r,f1



############### 改(一个分类准确率)
def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data, model_plotter):
	anomaly_correct, class_correct, class_total = 0, 0, 0
	tpl, tnl, fpl, fnl = [], [], [], []
	print('class_data', class_data)
	print('anomaly_data', anomaly_data)
	for i, d in enumerate(train_time_data):
		output = model(train_time_data[i], train_schedule_data[i])

		source_anomaly, source_prototype = output
		# print('outputttttt', output)
		# print('outputttttt', source_anomaly)
		# print('source_prototype', source_prototype)
		pred_state, prototypes = model(train_time_data[i], train_schedule_data[i])  #########

		# model_plotter.update_lines(pred_state.view(-1), train_time_data[i][-1])    ############

		# 计算异常准确度
		res, tp, tn, fp, fn = anomaly_accuracy(source_anomaly, anomaly_data[i], model_plotter)
		anomaly_correct += res
		tpl.append(tp)
		tnl.append(tn)
		fpl.append(fp)
		fnl.append(fn)

		tp += res;       #################
		fp += res;        #################
		tn += (1 - res);     #################
		fn += (1 - res)      #################

		# 分类准确度
		if np.sum(anomaly_data[i]) > 0:
			class_total += 1
			class_correct += class_accuracy(source_prototype, anomaly_data[i], class_data[i], model, model_plotter)
	tp, fp, tn, fn = np.mean(tpl), np.mean(fpl), np.mean(tnl), np.mean(fnl)

	########### 打印 tp, fp, tn, fn 的值
	tqdm.write(f'tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}')

	# 计算 precision, recall 和 f1 score
	precision = tp / (tp + fp) if (tp + fp) != 0 else 0
	recall = tp / (tp + fn) if (tp + fn) != 0 else 0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

	tqdm.write(f'P = {precision}, R = {recall}, F1 = {f1}')

	return anomaly_correct / len(train_time_data), class_correct / class_total, precision, recall, f1


# def class_accuracy(source_prototype, target_anomaly, target_class, model, model_plotter):
# 	correct, total = 0, 1e-4
# 	prototypes = []
#
# 	# 为每个类别记录 TP、FP、TN、FN
# 	class_counts = {0: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
# 					1: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
# 					2: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
#
# 	for i, sp in enumerate(source_prototype):
# 		if target_anomaly[i] > 0:  # 只对异常位置计算分类
# 			total += 1
# 			true_class = target_class[i]
# 			positive_loss = mse_loss(sp, model.prototype[true_class])
# 			negative_class_list = [0, 1, 2]
# 			negative_class_list.remove(true_class)
# 			negative_loss = [mse_loss(sp, model.prototype[nc]) for nc in negative_class_list]
#
# 			# 预测类别：正损失最小的类别
# 			pred_class = true_class if positive_loss <= min(negative_loss) else (
# 				0 if negative_loss[0] < negative_loss[1] else 2)
#
# 			# 更新 TP、FP、TN、FN
# 			if pred_class == true_class:
# 				class_counts[true_class]['tp'] += 1  # 真正例
# 				for nc in negative_class_list:
# 					class_counts[nc]['tn'] += 1  # 真负例
# 			else:
# 				class_counts[true_class]['fn'] += 1  # 假负例
# 				class_counts[pred_class]['fp'] += 1  # 假正例
# 				class_counts[negative_class_list[0 if pred_class == negative_class_list[1] else 1]][
# 					'tn'] += 1  # 另一个负类的真负例
#
# 			if pred_class == true_class:
# 				correct += 1
# 			prototypes.append((sp, true_class))
#
# 	if model_plotter is not None:
# 		model_plotter.update_class(prototypes, correct / total)
#
# 	return correct / total, class_counts


# def mse_loss(a, b):
# 	return np.mean((np.array(a) - np.array(b)) ** 2)




#############改(三个分类准确率)
# def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data, model_plotter):
#     anomaly_correct, class_correct, class_total = 0, 0, 0
#     tpl, tnl, fpl, fnl = [], [], [], []
#     class_metrics = {0: {'tp': [], 'fp': [], 'tn': [], 'fn': []},
#                      1: {'tp': [], 'fp': [], 'tn': [], 'fn': []},
#                      2: {'tp': [], 'fp': [], 'tn': [], 'fn': []}}
#
#     for i, d in enumerate(train_time_data):
#         output = model(train_time_data[i], train_schedule_data[i])
#         source_anomaly, source_prototype = output
#         pred_state, prototypes = model(train_time_data[i], train_schedule_data[i])
#
#         # 异常检测
#         res, tp, tn, fp, fn = anomaly_accuracy(source_anomaly, anomaly_data[i], model_plotter)
#         anomaly_correct += res
#         tpl.append(tp)
#         tnl.append(tn)
#         fpl.append(fp)
#         fnl.append(fn)
#
#         # 分类任务
#         if np.sum(anomaly_data[i]) > 0:
#             class_total += 1
#             acc, class_counts = class_accuracy(source_prototype, anomaly_data[i], class_data[i], model, model_plotter)
#             class_correct += acc
#             for cls in class_metrics:
#                 class_metrics[cls]['tp'].append(class_counts[cls]['tp'])
#                 class_metrics[cls]['fp'].append(class_counts[cls]['fp'])
#                 class_metrics[cls]['tn'].append(class_counts[cls]['tn'])
#                 class_metrics[cls]['fn'].append(class_counts[cls]['fn'])
#
#     # 循环结束后计算异常检测的总体指标
#     tp, fp, tn, fn = np.mean(tpl), np.mean(fpl), np.mean(tnl), np.mean(fnl)
#     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) != 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
#     tqdm.write(f"\nOverall Anomaly Detection:")
#     tqdm.write(f"tp = {tp:.2f}, fp = {fp:.2f}, tn = {tn:.2f}, fn = {fn:.2f}")
#     tqdm.write(f"Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")
#
#     # 循环结束后计算分类任务的每个类别指标
#     tqdm.write(f"\nClassification Metrics (After all samples):")
#     for cls in class_metrics:
#         tp = np.sum(class_metrics[cls]['tp'])  # 总和反映所有样本的计数
#         fp = np.sum(class_metrics[cls]['fp'])
#         tn = np.sum(class_metrics[cls]['tn'])
#         fn = np.sum(class_metrics[cls]['fn'])
#         precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) != 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
#         tqdm.write(f"Class {cls}:")
#         tqdm.write(f"  tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}")
#         tqdm.write(f"  Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")
#
#     return anomaly_correct / len(train_time_data), class_correct / class_total, precision, recall, f1


# def class_accuracy(source_prototype, target_anomaly, target_class, model, model_plotter):
# 	correct, total = 0, 1e-4
# 	prototypes = []
#
# 	# 为每个类别记录 TP、FP、TN、FN
# 	class_counts = {0: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
# 					1: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
# 					2: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
# 	#print('sdddddgghghjj',model)
# 	for i, sp in enumerate(source_prototype):
# 		if target_anomaly[i] > 0:  # 只对异常位置计算分类
# 			total += 1
# 			true_class = target_class[i]
# 			positive_loss = mse_loss(sp, model.prototype[true_class])
# 			negative_class_list = [0, 1, 2]
# 			negative_class_list.remove(true_class)
# 			negative_loss = [mse_loss(sp, model.prototype[nc]) for nc in negative_class_list]
#
# 			# 预测类别：正损失最小的类别
# 			pred_class = true_class if positive_loss <= min(negative_loss) else (
# 				0 if negative_loss[0] < negative_loss[1] else 2)
#
# 			# 更新 TP、FP、TN、FN
# 			if pred_class == true_class:
# 				class_counts[true_class]['tp'] += 1  # 真正例
# 				for nc in negative_class_list:
# 					class_counts[nc]['tn'] += 1  # 真负例
# 			else:
# 				class_counts[true_class]['fn'] += 1  # 假负例
# 				class_counts[pred_class]['fp'] += 1  # 假正例
# 				class_counts[negative_class_list[0 if pred_class == negative_class_list[1] else 1]][
# 					'tn'] += 1  # 另一个负类的真负例
#
# 			if pred_class == true_class:
# 				correct += 1
# 			prototypes.append((sp, true_class))
#
# 	if model_plotter is not None:
# 		model_plotter.update_class(prototypes, correct / total)
#
# 	return correct / total, class_counts
