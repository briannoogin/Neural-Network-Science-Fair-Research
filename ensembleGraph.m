data = xlsread('ensemblenumber.xlsx');
hold on
plot(data(:,1),data(:,2));
set(gca,'fontsize',14);
title('Ensemble Size Performance','FontSize',18);
xlabel('Ensemble Size','FontSize',18);
ylabel('Area under the Curve','FontSize',18);
