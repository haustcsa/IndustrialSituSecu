% SEIR模型
clear;clc;tic;
% 参数设置
N = 1000;             % 节点数量
m0 = 5;               % BA模型初始节点数量
m = 3;                % 每个新节点连接的边数
T = 100;              % 仿真时间步长
beta = 0.3;           % 感染率（固定概率）
sigma = 0.2;          % 潜伏期转换为感染期的速率
gamma = 0.1;          % 康复率
initial_infected_ratio = 0.1; % 初始感染者的比例（例如，10%）
num_runs = 10;        % 仿真重复次数

rng(0);  % 设置随机数种子为0 确保每次运行代码时，生成的随机数序列是相同的

% 初始化用于累积结果的数组
S_total = zeros(T, 1);
E_total = zeros(T, 1);
I_total = zeros(T, 1);
R_total = zeros(T, 1);


% 初始化用于存储净变化量的累积数组
delta_S_total = zeros(T, 1);
delta_E_total = zeros(T, 1);
delta_I_total = zeros(T, 1);
delta_R_total = zeros(T, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% ！如果你有自己的网络矩阵，则无需生成BA网络，直接将 A = your_adj_mattrix;
%% 生成无标度网络（Barabási-Albert模型）
% 初始化邻接矩阵
A = zeros(N);

% 构建初始完全图
A(1:m0, 1:m0) = ones(m0) - eye(m0);
nodeDegrees = sum(A, 2);

% 添加剩余的节点
for n = (m0+1):N
    % 仅使用已有节点的度数
    prob = nodeDegrees(1:n-1) / sum(nodeDegrees(1:n-1));
    connectedNodes = randsample(n-1, m, true, prob);
    A(n, connectedNodes) = 1;
    A(connectedNodes, n) = 1;
    nodeDegrees(n) = m;
    nodeDegrees(connectedNodes) = nodeDegrees(connectedNodes) + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 为每个节点分配先验概率值（0~1之间）
prior_probs = rand(N, 1);   %%%%%%%%%%%%%%%%%%%%%% !替换为你自己的先验概率

% 规范化先验概率，使其和为1
prior_probs = prior_probs / sum(prior_probs);

% 根据先验概率选择初始感染者
num_initial_infected = round(initial_infected_ratio * N);

for run = 1:num_runs
    
    % 使用 datasample 函数进行加权无放回抽样
    initialInfected = datasample(1:N, num_initial_infected, 'Replace', false, 'Weights', prior_probs);
    
    % 初始化节点状态：1=易感者（S），2=潜伏者（E），3=感染者（I），4=恢复者（R）
    states = ones(N, 1);
    
    % 设置初始感染者的状态为感染者（I）
    states(initialInfected) = 3;
    
    % 记录每个时间步长中各状态的人数
    S_history = zeros(T, 1);
    E_history = zeros(T, 1);
    I_history = zeros(T, 1);
    R_history = zeros(T, 1);
    
    % 初始状态计数
    S_history(1) = sum(states == 1);
    E_history(1) = sum(states == 2);
    I_history(1) = sum(states == 3);
    R_history(1) = sum(states == 4);

    % 临时存储本次仿真的净变化量
    delta_S = zeros(T, 1);
    delta_E = zeros(T, 1);
    delta_I = zeros(T, 1);
    delta_R = zeros(T, 1);

    for t = 2:T  % 从t=2开始，便于计算变化量
        % 记录上一个时间步各状态的人数
        prev_S = S_history(t - 1);
        prev_E = E_history(t - 1);
        prev_I = I_history(t - 1);
        prev_R = R_history(t - 1);
        
        newStates = states;
        for i = 1:N
            if states(i) == 1 % 易感者（S）
                neighbors = find(A(i, :) == 1);
                infectedNeighbors = states(neighbors) == 3;
                if any(infectedNeighbors)
                    if rand < beta
                        newStates(i) = 2; % 转为潜伏者（E）
                    end
                end
            elseif states(i) == 2 % 潜伏者（E）
                if rand < sigma
                    newStates(i) = 3; % 转为感染者（I）
                end
            elseif states(i) == 3 % 感染者（I）
                if rand < gamma
                    newStates(i) = 4; % 转为恢复者（R）
                end
            end
        end
        states = newStates;

        % 记录当前状态的人数
        S_history(t) = sum(states == 1);
        E_history(t) = sum(states == 2);
        I_history(t) = sum(states == 3);
        R_history(t) = sum(states == 4);

         % 计算每个状态的净变化量
        delta_S(t) = S_history(t) - prev_S;
        delta_E(t) = E_history(t) - prev_E;
        delta_I(t) = I_history(t) - prev_I;
        delta_R(t) = R_history(t) - prev_R;

    end
    % 累加每次仿真的结果
    S_total = S_total + S_history;
    E_total = E_total + E_history;
    I_total = I_total + I_history;
    R_total = R_total + R_history;

    % 累加每次仿真的净变化量
    delta_S_total = delta_S_total + delta_S;
    delta_E_total = delta_E_total + delta_E;
    delta_I_total = delta_I_total + delta_I;
    delta_R_total = delta_R_total + delta_R;
end

% 计算平均值
S_avg = S_total / num_runs;
E_avg = E_total / num_runs;
I_avg = I_total / num_runs;
R_avg = R_total / num_runs;

% 计算净变化量的平均值
delta_S_avg = delta_S_total / num_runs;
delta_E_avg = delta_E_total / num_runs;
delta_I_avg = delta_I_total / num_runs;
delta_R_avg = delta_R_total / num_runs;

% 计算比例
S_ratio = S_avg / N;
E_ratio = E_avg / N;
I_ratio = I_avg / N;
R_ratio = R_avg / N;

% 计算比例
delta_S_ratio = abs(delta_S_avg / N);
delta_E_ratio = abs(delta_E_avg / N);
delta_I_ratio = abs(delta_I_avg / N);
delta_R_ratio = abs(delta_R_avg / N);

% 绘制平均净变化量
figure;
plot(2:T, delta_S_ratio(2:T), 'b', 2:T, delta_E_ratio(2:T), 'y', 2:T, delta_I_ratio(2:T), 'r', 2:T, delta_R_ratio(2:T), 'g', 'LineWidth', 1.5);
legend('ΔS', 'ΔE', 'ΔI', 'ΔR');
xlabel('t');
ylabel('prob');
title('Posterior  prob');
grid on;

% 绘制平均结果（比例）
figure;
main_ax = gca;
plot(1:T, S_ratio, 'b', 1:T, E_ratio, 'y', 1:T, I_ratio, 'r', 1:T, R_ratio, 'g', 'LineWidth', 2);
legend('S', 'E', 'I', 'R');
xlabel('t');
ylabel('prop');
title([' \beta = ', sprintf('%.2f', beta)]);
grid on;


% 添加放大图
% 选择需要放大的区域，例如时间步长为 20 到 40 的部分
zoom_start = 10;
zoom_end = 30;

% 在插图中动态设置X和Y轴范围
I_zoom_data = I_ratio(zoom_start:zoom_end); % 获取插图中的I态数据
zoom_ylim = [min(I_zoom_data), max(I_zoom_data)]; % 计算动态Y轴范围

% 创建一个小的坐标轴用于插图
axes_position = [0.6, 0.2, 0.25, 0.25]; % [left, bottom, width, height]
axes_zoom = axes('Position', axes_position);

% 在插图上绘制感染者（I）状态的数据
plot(zoom_start:zoom_end, I_zoom_data, 'r', 'LineWidth', 1.5);
ylim(zoom_ylim);
xlim([zoom_start, zoom_end]);

% 绘制箭头
% 找到主图和插图中的感染者（I）曲线峰值位置
[I_peak_value, I_peak_idx] = max(I_ratio); % 主图峰值
[I_inset_peak_value, I_inset_peak_idx] = max(I_ratio(zoom_start:zoom_end)); % 插图峰值
I_inset_peak_idx = I_inset_peak_idx + zoom_start - 1; % 插图峰值实际索引

title(sprintf('(%d, %.4f)', I_inset_peak_idx, I_inset_peak_value));

% 将峰值的数据坐标转换为图形坐标
% 获取主图和插图的位置
main_pos = get(main_ax, 'Position'); % [left, bottom, width, height]
zoom_pos = axes_position;            % [left, bottom, width, height]

% 获取主图的X和Y轴范围
main_xlim = get(main_ax, 'XLim');
main_ylim = get(main_ax, 'YLim');

% 计算主图峰值的规范化坐标
x_norm_main = (I_peak_idx - main_xlim(1)) / (main_xlim(2) - main_xlim(1));
y_norm_main = (I_peak_value - main_ylim(1)) / (main_ylim(2) - main_ylim(1));
peak_x_fig_main = main_pos(1) + x_norm_main * main_pos(3);
peak_y_fig_main = main_pos(2) + y_norm_main * main_pos(4);

% 计算插图峰值的规范化坐标
x_norm_inset = (I_inset_peak_idx - zoom_start) / (zoom_end - zoom_start);
y_norm_inset = (I_inset_peak_value - zoom_ylim(1)) / (zoom_ylim(2) - zoom_ylim(1));
peak_x_fig_inset = zoom_pos(1) + x_norm_inset * zoom_pos(3);
peak_y_fig_inset = zoom_pos(2) + y_norm_inset * zoom_pos(4);

% 添加箭头，从主图的峰值指向插图的峰值
annotation('arrow', [peak_x_fig_main, peak_x_fig_inset], [peak_y_fig_main, peak_y_fig_inset]);

toc;