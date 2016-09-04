fileName = 'breastCancerFullDataSet.xlsx';
[numeric,text,excel] = xlsread(fileName);
targetVector = numeric(:,1);
inputMatrix = numeric(:,2:10);
net = feedforwardnet;
