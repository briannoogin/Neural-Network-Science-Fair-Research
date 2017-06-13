[data,~,excel] = xlsread('breastKaggleFixed');
dataNoOutputVariables = data(:,2:size(data,2));
excelNoOutputVariables = excel(:,2:size(excel,2));
%[fixed,settings] = fixunknowns(data.');
%fixed = fixed.';
corMatrix = corrcoef(data);
corMatriz2 = corrcoef(dataNoOutputVariables);

