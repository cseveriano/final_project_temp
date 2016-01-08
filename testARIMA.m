function [result] = testARIMA(TestData, index)
TV.T = TestData.output(index)';
TV.P = TestData.input(index,:)';

model = arima(0,1,1);
fit = estimate(model,TV.P,'Display','off');
[TY,YMSE] = forecast(fit,1);

result.TestingAccuracyRMSE=sqrt(YMSE);            %   Calculate testing accuracy (RMSE) for regression case
result.TestingAccuracyPercent=percent(TV.T - TY);

end
