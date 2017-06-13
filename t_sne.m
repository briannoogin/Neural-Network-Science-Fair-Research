[trainNumeric,text,excel] = xlsread('breastCancerKaggleQuestionMarks');
data = trainNumeric(:,2:size(trainNumeric,2));
output = trainNumeric(:,1);
text = ['benign','maligant'];
t = tsne(data,output,2,2,40);
gscatter(t(:,1),t(:,2),output,'rb');
xlabel('T-SNE X','FontSize',18);
ylabel('T-SNE Y','FontSize',18);
set(gca,'fontsize',18);
title('T-SNE Plot');
t = cat(2,output,t);
xlswrite('tsneOutput.xlsx',t);