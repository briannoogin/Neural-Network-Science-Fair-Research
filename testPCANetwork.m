% Reads Excel File
trainingfileName = 'mammographBreastCancerWithDeletedData.xlsx';
[trainNumeric,text,excel] = xlsread(trainingfileName);
testfileName = 'mammographBreastCancerWithDeletedData.xlsx';
[testNumeric,testText,testExcel] = xlsread(testfileName);
% Gets the data targets
trainTargetVector = trainNumeric(:,1);
testTargetVector = testNumeric(:,1);
% Gets the data
trainingInputMatrix = trainNumeric(:,2:size(trainNumeric,2));
testInputMatrix = testNumeric(:,2:size(testNumeric,2));
% Creates network with three layers
net = feedforwardnet(10);
net.numlayers = 4;
% Tranpose data so it works with the toolbox
trainTargetVector = trainTargetVector.';
trainingInputMatrix = trainingInputMatrix.';
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
net.performParam.regularization = 0;
net.trainFcn = 'trainscg';
net.initFcn = 'initlay';
net.performFcn = 'mse';
% Change train settings and train network 
net.trainParam.showWindow = 0;
trialPercentPerformance = zeros(11,1);
trialTrainPerformance = zeros(11,1);
error = crossValidation(net,10,'pcaOutput');