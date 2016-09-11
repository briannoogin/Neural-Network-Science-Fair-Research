function [y,targetVector] = linearNetwork
% Reads Excel File
trainingfileName = 'scienceFairBreastCancerData.xls';
[trainNumeric,text,excel] = xlsread(trainingfileName);
testfileName = 'breastTestData.xlsx';
[testNumeric,testText,testExcel] = xlsread(testfileName);
trainTargetVector = trainNumeric(:,1);
testTargetVector = testNumeric(:,1);
% Gets the data
trainingInputMatrix = trainNumeric(:,2:size(excel,2));
testInputMatrix = testNumeric(:,2:size(excel,2));
% Creates network with three layers
net = feedforwardnet(1);
net.numlayers = 2;
% Tranpose data so it works with the toolbox
trainTargetVector = trainTargetVector.';
trainingInputMatrix = trainingInputMatrix.';
testTargetVector = testTargetVector.';
testInputMatrix = testInputMatrix.';
% Designate the activation functions 
net.layers{1,1}.transferfcn = 'tansig';
net.layers{2,1}.transferfcn = 'tansig';
% Connects layers and bias units
net.layerConnect = [0 0 ; 1 0 ];
net.biasConnect = [1;1;];
net.outputConnect = [0 1];
net = configure(net,trainingInputMatrix,trainTargetVector);
% Train
[net,record] = train(net,trainingInputMatrix,trainTargetVector,'useGPU','yes');
y = net(trainingInputMatrix);
y = y > .5;
targetVector = trainTargetVector;
view(net);