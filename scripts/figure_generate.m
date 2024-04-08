% Performance Evaluation

Rate_user_num = zeros(8,6);
Rate_ap_num = zeros(8,6);
Rate_max_power = zeros(8,6);
Rate_cluster_size = zeros(8,6);
Delay_user_num = zeros(8,6);
Delay_ap_num = zeros(8,6);
Delay_max_power = zeros(8,6);
Delay_cluster_size = zeros(8,6);

% Figure 1: Convergence Performance
reward_list = zeros(8,250);
reward_list(1,:) = reward_UCMEC_coop;
reward_list(2,:) = reward_UCMEC_noncoop;
reward_list(3,:) = reward_CBO_coop;
reward_list(4,:) = reward_CBO_noncoop;
reward_list(5,:) = reward_MPO_coop;
reward_list(6,:) = reward_MPO_noncoop;
reward_list(7,:) = reward_MADDPG_coop;
reward_list(8,:) = reward_IQL_noncoop;

step_index = 1:200:50000;
load("reward_list_processed.mat");

reward_list = reward_list * 1000;
reward_list_processed = reward_list_processed * 1000;
% plot(step_index,reward_list(1,:),'linewidth',2,'Color',[0  114  189]/255); hold on;
% plot(step_index,reward_list(2,:),'linewidth',2,'Color',[0.3010 0.7450 0.9330]); 
% plot(step_index,reward_list(3,:),'linewidth',2,'Color',[0.8500 0.3250 0.0980]); 
% plot(step_index,reward_list(4,:),'linewidth',2,'Color',[0.9290 0.6940 0.1250]); 
% plot(step_index,reward_list(5,:),'linewidth',2,'Color',[0.4660 0.6740 0.1880]); 
% plot(step_index,reward_list(6,:),'linewidth',2,'Color',[0.6784  1  0.1843]); 
% plot(step_index,reward_list(7,:),'linewidth',2,'Color',[0.4940 0.1840 0.5560]); 
% plot(step_index,reward_list(8,:),'linewidth',2,'Color',[0.6350 0.0780 0.1840]); 

plot(step_index,reward_list_processed(1,:),'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(step_index,reward_list_processed(2,:),'linewidth',2,'Color',[0.3010 0.7450 0.9330]); 
plot(step_index,reward_list_processed(3,:),'linewidth',2,'Color',[0.8500 0.3250 0.0980]); 
plot(step_index,reward_list_processed(4,:),'linewidth',2,'Color',[0.9290 0.6940 0.1250]); 
plot(step_index,reward_list_processed(5,:),'linewidth',2,'Color',[0.4660 0.6740 0.1880]); 
plot(step_index,reward_list_processed(6,:),'linewidth',2,'Color',[0.6784  1  0.1843]); 
plot(step_index,reward_list_processed(7,:),'linewidth',2,'Color',[0.4940 0.1840 0.5560]); 
plot(step_index,reward_list_processed(8,:),'linewidth',2,'Color',[0.6350 0.0780 0.1840]); 

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Episode Index'),ylabel('Reward');
set(gca,'FontName','Times New Roman','FontSize',12);

% Figure 2: Training Time
% time_list = zeros(8,1);
figure('visible','on')
time_list_multi = [1200,1073;1140,1013;1018,905;1753 0;0,1093];
b = bar(time_list_multi,1,'EdgeColor','k','LineWidth',1);
hatchfill2(b(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
legend([b(1),b(2)],'Coop','Non-Coop');
ax = gca;
ax.XTickLabel = {'Proposed','CBO','MPO','MADDPG','IQL'};
grid on;
set(gca,'FontName','Times New Roman','FontSize',12);

% Figure 3: User Num VS Uplink Rate
User_index = [5,10,15,20,25,30];

plot(User_index,Rate_user_num(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(User_index,Rate_user_num(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(User_index,Rate_user_num(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(User_index,Rate_user_num(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(User_index,Rate_user_num(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(User_index,Rate_user_num(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(User_index,Rate_user_num(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(User_index,Rate_user_num(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Number of Users'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',12);


% Figure 4: AP Num VS Uplink Rate
AP_Num_index = [10,20,30,40,50,60];

plot(AP_Num_index,Rate_ap_num(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(AP_Num_index,Rate_ap_num(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(AP_Num_index,Rate_ap_num(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(AP_Num_index,Rate_ap_num(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(AP_Num_index,Rate_ap_num(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(AP_Num_index,Rate_ap_num(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(AP_Num_index,Rate_ap_num(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(AP_Num_index,Rate_ap_num(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Number of APs'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',12);

% Figure 5: Maximum Power VS Uplink Rate
Max_power_index = [0.05,0.1,0.15,0.2,0.25,0.3];

plot(Max_power_index,Rate_max_power(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(Max_power_index,Rate_max_power(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(Max_power_index,Rate_max_power(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(Max_power_index,Rate_max_power(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(Max_power_index,Rate_max_power(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(Max_power_index,Rate_max_power(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(Max_power_index,Rate_max_power(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(Max_power_index,Rate_max_power(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Maximum Transmit Power (W)'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',12);

% Figure 6: AP Cluster Size VS Uplink Rate
AP_cluster_index = [1,2,3,4,5,6];

plot(AP_cluster_index,Rate_cluster_size(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(AP_cluster_index,Rate_cluster_size(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(AP_cluster_index,Rate_cluster_size(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(AP_cluster_index,Rate_cluster_size(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(AP_cluster_index,Rate_cluster_size(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(AP_cluster_index,Rate_cluster_size(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(AP_cluster_index,Rate_cluster_size(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(AP_cluster_index,Rate_cluster_size(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('AP Cluster Size'),ylabel('Average Uplink Rate (Mbps)');
set(gca,'FontName','Times New Roman','FontSize',12);



% Figure 7: User Num VS Average Total Delay
User_index = [5,10,15,20,25,30];

plot(User_index,Delay_user_num(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(User_index,Delay_user_num(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(User_index,Delay_user_num(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(User_index,Delay_user_num(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(User_index,Delay_user_num(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(User_index,Delay_user_num(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(User_index,Delay_user_num(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(User_index,Delay_user_num(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Number of Users'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',12);


% Figure 8: AP Num VS Average Total Delay
AP_Num_index = [10,20,30,40,50,60];

plot(AP_Num_index,Delay_ap_num(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(AP_Num_index,Delay_ap_num(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(AP_Num_index,Delay_ap_num(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(AP_Num_index,Delay_ap_num(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(AP_Num_index,Delay_ap_num(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(AP_Num_index,Delay_ap_num(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(AP_Num_index,Delay_ap_num(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(AP_Num_index,Delay_ap_num(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Number of APs'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',12);

% Figure 9: Maximum Power VS Average Total Delay
Max_power_index = [0.05,0.1,0.15,0.2,0.25,0.3];

plot(Max_power_index,Delay_max_power(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(Max_power_index,Delay_max_power(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(Max_power_index,Delay_max_power(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(Max_power_index,Delay_max_power(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(Max_power_index,Delay_max_power(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(Max_power_index,Delay_max_power(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(Max_power_index,Delay_max_power(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(Max_power_index,Delay_max_power(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('Maximum Transmit Power (W)'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',12);

% Figure 10: AP Cluster Size VS Average Total Delay
AP_cluster_index = [1,2,3,4,5,6];

plot(AP_cluster_index,Delay_cluster_size(1,:),"-o",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255); hold on;
plot(AP_cluster_index,Delay_cluster_size(2,:),"o--",'Markersize',7,'linewidth',2,'Color',[0  114  189]/255);

plot(AP_cluster_index,Delay_cluster_size(3,:),"-d",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255); 
plot(AP_cluster_index,Delay_cluster_size(4,:),"d--",'Markersize',7,'linewidth',2,'Color',[217  83  25]/255);

plot(AP_cluster_index,Delay_cluster_size(5,:),"-*",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);
plot(AP_cluster_index,Delay_cluster_size(6,:),"*--",'Markersize',7,'linewidth',2,'Color',[237  177  32]/255);

plot(AP_cluster_index,Delay_cluster_size(7,:),"-x",'Markersize',7,'linewidth',2,'Color',[126 47 142]/255);
plot(AP_cluster_index,Delay_cluster_size(8,:),"-^",'Markersize',7,'linewidth',2,'Color',[0.13 0.55 0.13]);

grid on;
legend('Proposed (Coop)','Proposed (Non-Coop)', "CBO (Coop)", "CBO (Non-Coop)", "MPO (Coop)", "MPO (Non-Coop)", "MADDPG (Coop)", "IQL (Non-Coop)");
xlabel('AP Cluster Size'),ylabel('Average Total Delay (ms)');
set(gca,'FontName','Times New Roman','FontSize',12);



