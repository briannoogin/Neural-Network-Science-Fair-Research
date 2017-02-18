[trainNumeric,text,excel] = xlsread('andrewsplot');
group = excel(2:size(excel,1),1);
andrewsplot(trainNumeric,'Standardize','on','group',group);