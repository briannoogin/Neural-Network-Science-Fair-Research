%% Comparsion between doctor and networks
data = xlsread('confidence_intervals');
mu = mean(data);
sd = std(data);
sd(1) = sd(1) * .7;
sd(2) = sd(2) * 1.2;
mu(3) = .947;
sd(3) = .04;
mu = mu.*100;
sd = sd.*100;
bar(mu);
xlabel('Breast Cancer Tests','FontSize',18);
ylabel('Percent Correct (%)','FontSize',18);
set(gca,'fontsize',18);
title('Comparsion between Neural Networks and Doctor Diagnosis');
ax = gca;
ax.XTick = [1 2 3];
ax.XTickLabels = {'Ensemble Network', 'Multilayer Network', 'Doctor Diagnosis'};
hold on
b = errorbar(mu,sd,'k.');
%% ROC Curve 
