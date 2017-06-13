[trainNumeric,text,excel] = xlsread('breastCancerKaggleQuestionMarks');
data = trainNumeric(:,2:size(trainNumeric,2));
output = trainNumeric(:,1);
[coef,score,latent,tsquared,explained,mu] = pca(data,'NumComponents',2);
%plot(explained);
totalVariance = explained(1,1);
% Computes the variance explained by adding each component so elbow method
% can be used
for variance = 2:size(explained,1)
    totalVariance(variance) = totalVariance(variance - 1) + explained(variance);
end
%% Variance Graph
plot(totalVariance);
xlabel('Number of Components','FontSize',18);
ylabel('Variance Accounted (%)','FontSize',18);
set(gca,'fontsize',18);
title('PCA Components Variances');
pcaOutput = cat(2,output,score);
%xlswrite('pcaOutput.xlsx',pcaOutput);

%% Scatter Plot
%{
gscatter(score(:,1),score(:,2),output,'rb');
xlabel('PCA Score 1','FontSize',18);
ylabel('PCA Score 2','FontSize',18);
set(gca,'fontsize',18);
title('PCA Scatterplot');
%}


