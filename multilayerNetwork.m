% Reads Excel File
trainingfileName = 'breastCancerFullDataSet';
[trainNumeric,text,excel] = xlsread(trainingfileName);
testfileName = 'breastCancerFullDataSet';
[testNumeric,testText,testExcel] = xlsread(testfileName);
% Autoencode the data
encodeData = trainNumeric(:,2:size(trainNumeric,2));
encodeData = encodeData.';
autoenc = trainAutoencoder(encodeData,5);
features = encode(autoenc,encodeData);
% Gets the data targets
trainTargetVector = trainNumeric(:,1);
testTargetVector = testNumeric(:,1);
% Gets the data
trainingInputMatrix = features;
testInputMatrix = testNumeric(:,2:size(testNumeric,2));
% Creates network with three layers
net = feedforwardnet(10);
net.numlayers = 4;
% Tranpose data so it works with the toolbox
trainTargetVector = trainTargetVector.';
testTargetVector = testTargetVector.';
testInputMatrix = testInputMatrix.';
% Name the layers
net.layers{1}.name = 'Hidden Layer 1';
net.layers{2}.name = 'Hidden Layer 2';
net.layers{3}.name = 'Hidden Layer 3';
net.layers{4}.name = 'Output';
% Designate the activation functions and number of neural units
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'logsig';
net.layers{4}.transferFcn = 'softmax';
net.layers{1,1}.size = 100;
net.layers{2,1}.size = 50;
net.layers{3,1}.size = 25;
% Connects layers and bias units
net.layerConnect = [0 0 0 0; 1 0 0 0; 0 1 0 0;0 0 1 0];
net.biasConnect = [1;1;1;1];
net.outputConnect = [0 0 0 1];
% Shape the network to the data
%net = configure(net,trainingInputMatrix,trainTargetVector);
net.performParam.regularization = .9;
net.trainFcn = 'trainscg';
net.initFcn = 'initlay';
net.layers{1}.netInputFcn = 'netsum';
net.layers{2}.netInputFcn = 'netsum';
net.layers{3}.netInputFcn = 'netsum';
% Change train settings and train network 
net.trainParam.showWindow = 1;
trialPercentPerformance = zeros(10,1);
trialTrainPerformance = zeros(10,1);
% Do multiple trials for percent correct
performanceStruct = struct('Net',0);
for networkCount = 1 : 10
net = init(net);
[net1,record] = train(net,trainingInputMatrix,trainTargetVector,'useGPU','yes');
trainPerformance = perform(net, trainingInputMatrix,trainTargetVector);
% Calculating percent correct
y = net1(trainingInputMatrix);
% Changes the probability boundary
y = y > .5;
findCorrect = find(y == trainTargetVector);
percentCorrect = size(findCorrect,2) / size(trainTargetVector,2); 
trialPercentPerformance(networkCount) = percentCorrect;
trialTrainPerformance(networkCount) = trainPerformance;
performanceStruct(networkCount).Net = net1;
end
[~,bestPerformanceIndex] = max(trialPercentPerformance);
net = performanceStruct(bestPerformanceIndex).Net;
% Calculate the average performance
percentCorrect = sum(trialPercentPerformance,1) / size(trialPercentPerformance,1);
trainPerformance = sum(trialPercentPerformance,1) / size(trialPercentPerformance,1);
% Test Performance
testPerformance = perform(net,testInputMatrix,testTargetVector);
truePositivesAndFalsePositives = roc(trainTargetVector,y);
%plotroc(trainTargetVector,y);
% Test linear performance
%linearY = linearNetwork;
%plotroc(trainTargetVector,linearY);
%%%% Outlier System Idea
%{
I should look at the data and see what examples were wrong the most often.
Hopefully, the problem is that one feature is a problem so I can easily put
the outliers in an outlier dataset.
%}
%view(net);
%plot(trialPercentPerformance);
plotconfusion(trainTargetVector,y);
% Finds where the network has failed
error = find(~y == trainTargetVector);
% Finds all the errors and collects them in a matrix
errorMatrix = cell(size(error,2) + 1,size(excel,2));
errorMatrix(1,:) = excel(1,:);
trainNumeric = num2cell(trainNumeric);
for errorIndex = 1 : size(error,2)
    errorMatrix(errorIndex + 1,:) = trainNumeric(error(1,errorIndex),:);
end
net
view(net);
swag = net.LW;
layers = net.layers;
