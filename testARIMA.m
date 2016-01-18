function [result] = testARIMA(ArimaModel, TestData, index)
TV.T = TestData.output(index)';
TV.P = TestData.input(index,:)';

[TY,YMSE] = forecast(ArimaModel,1, 'Y0', TV.P);

result.TestingAccuracyRMSE=sqrt(YMSE);            %   Calculate testing accuracy (RMSE) for regression case
result.TestingAccuracyPercent=percent(TV.T - TY);
result.forecast = TY;

end
