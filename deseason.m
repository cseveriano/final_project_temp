clear all;

trainPath =  'C:\Users\Carlos\Documents\Projetos Machine Learning\Multiagent\Data\Input\*\*\';

file_pattern = '*[AVG-30].txt';
window_size = 8;
nclusters = 8;
hidden_layer = 6;
kfold = 1;

% Read train files
files = rdir(strcat(trainPath, file_pattern));
train_raw_data = [];

for i = 1 : 6
    x = importdata(files(i).name,'\t', 1);
    train_raw_data = [train_raw_data x.data(:,1)'];
    clear x;
end

T = length(train_raw_data);
figure
plot(train_raw_data)
% h1 = gca;
% h1.XLim = [0,T];
% h1.XTick = 1:48:T;
% h1.XTickLabel = datestr(dates(1:48:T),10);
title '30min Irradiance';
ylabel 'GSi0';
hold on

%-------- Moving average ----%

sW13 = [1/96;repmat(1/48,47,1);1/96];
yS = conv(train_raw_data,sW13,'same');
yS(1:24) = yS(25); yS(T-23:T) = yS(T-24);

xt = train_raw_data-yS;

h = plot(yS,'r','LineWidth',2);
legend(h,'49-Term Moving Average')
hold off

%-------- Seasonal indices ----%

s = 48;
sidx = cell(s,1);
for i = 1:s
 sidx{i,1} = i:s:T;
end

%-------- Seasonal filter ----%
sst = cellfun(@(x) mean(xt(x)),sidx);


nc = floor(T/s); % no. complete days
rm = mod(T,s); % no. extra inputs
sst = [repmat(sst,nc,1);sst(1:rm)];

% Center the seasonal estimate (additive)
sBar = mean(sst); % for centering
sst = sst-sBar;

figure
plot(sst)
title 'Stable Seasonal Component';
