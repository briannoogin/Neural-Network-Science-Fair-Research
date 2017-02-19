[trainNumeric,text,excel] = xlsread('breastCancerKaggle.xlsx');
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
plot(totalVariance);
xlabel('Number of Components');
ylabel('Variance Accounted (%)');
title('PCA Components Variances');
pcaOutput = cat(2,output,score);
save pcaOutput

