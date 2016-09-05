% Reads Excel File
trainingfileName = 'breastTraining.xlsx';
[trainNumeric,text,excel] = xlsread(trainingfileName);
testfileName = 'breastTraining.xlsx';
[testNumeric,testText,testExcel] = xlsread(testfileName);
% Gets the data targets
trainTargetVector = trainNumeric(:,1);
testTargetVector = testNumeric(:,1);
% Gets the data
trainingInputMatrix = trainNumeric(:,2:10);
testInputMatrix = testNumeric(:,2:10);
% Creates network with three layers
net = feedforwardnet(10);
net.numlayers = 3;
% Tranpose data so it works with the toolbox
trainTargetVector = trainTargetVector.';
trainingInputMatrix = trainingInputMatrix.';
testTargetVector = testTargetVector.';
testInputMatrix = testInputMatrix.';
% Name the layers
net.layers{1}.name = 'Hidden Layer 1';
net.layers{2}.name = 'Hidden Layer 2';
net.layers{3}.name = 'Output';
% Designate the activation functions and number of neural units
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'logsig';
net.layers{1,1}.size = 50;
net.layers{2,1}.size = 25;
% Connects layers and bias units
net.layerConnect = [0 0 0; 1 0 0; 0 1 0];
net.biasConnect = [1;1;0];
net.outputConnect = [0 0 1];
% Shape the network to the data
net = configure(net,trainingInputMatrix,trainTargetVector);
net.performParam.regularization = .4;
% Train network 
[net,record] = train(net,trainingInputMatrix,trainTargetVector);
% Calculating percent correct
y = net(trainingInputMatrix);
y = y > .51;
percentCorrect = find(y == trainTargetVector);
percentCorrect = size(percentCorrect,2) / size(trainTargetVector,2);  
% Test Performance
testPerformance = perform(net,testInputMatrix,testTargetVector);
view(net); 
