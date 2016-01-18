function [deseason_data, season_component] = deseasonalize(original_data)

% Original data has one day per row. Here is converted to a single vector
y = reshape( original_data' ,1,numel(original_data));

T = length(y);
figure
plot(y)
% h1 = gca;
% h1.XLim = [0,T];
% h1.XTick = 1:48:T;
% h1.XTickLabel = datestr(dates(1:48:T),10);
title '30min Irradiance';
ylabel 'GSi0';
hold on

%-------- Moving average ----%

sW13 = [1/96;repmat(1/48,47,1);1/96];
yS = conv(y,sW13,'same');
yS(1:24) = yS(25); yS(T-23:T) = yS(T-24);

xt = y-yS;

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

season_component = vec2mat(sst,48);

deseason_data = original_data - season_component;
end


