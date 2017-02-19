[trainNumeric,text,excel] = xlsread('breastCancerKaggle.xlsx');
data = trainNumeric(:,2:size(trainNumeric,2));
output = trainNumeric(:,1);
text = ['benign','maligant'];
t = tsne(data,output,3,2,25);
gscatter(t(:,1),t(:,2),output);
xlabel('T-SNE X');
ylabel('T_SNE Y');
title('T-SNE Plot');
