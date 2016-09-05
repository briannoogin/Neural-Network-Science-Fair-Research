% Reads Excel File
fileName = 'breastCancerFullDataSet.xlsx';
[numeric,text,excel] = xlsread(fileName);
% Gets the data targets
targetVector = numeric(:,1);
% Gets the data
inputMatrix = numeric(:,2:10);
% Creates network with three layers
net = feedforwardnet(10);
net.numlayers = 3;
% Tranpose data so it works with the toolbox
targetVector = targetVector.';
inputMatrix = inputMatrix.';
% Name the layers
net.layers{1}.name = 'Hidden Layer 1';
net.layers{2}.name = 'Hidden Layer 2';
% Designate the activation functions
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'tansig';
% Connects layers and bias units
net.layerConnect = [0 0 0  ; 0 0 1; 1 0 0];
net.biasConnect = [1;1;1];
% Shape the network to the data
net = configure(net,inputMatrix,targetVector);
% Train network 
%net = train(net,inputMatrix,targetVector);
% Calculating percent correct
%y = net(inputMatrix);
y = y > .51;
percentCorrect = find(y == targetVector);
percentCorrect = size(percentCorrect,2) / size(targetVector,2); 
layers = net.layers;
view(net); 