fileName = 'breastCancerFullDataSet.xlsx';
[numeric,text,excel] = xlsread(fileName);
targetVector = numeric(:,1);
inputMatrix = numeric(:,2:10);
net = cascadeforwardnet(10);
net.numlayers = 3;
targetVector = targetVector.';
inputMatrix = inputMatrix.';
net = configure(net,inputMatrix,targetVector);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
net.inputConnect(1,1) = 1
net.inputConnect(2,1) = 1;
net.inputConnect(3,1) = 1;
net.layers
net = train(net,inputMatrix,targetVector);
view(net);