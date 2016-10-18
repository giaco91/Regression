function [NSE target_estimation]= cross_validation(X_val,y_val,betahat,y_mean_train,x_mean_train)
n=size(X_val,1);
for i=1:n
    X_val(i,:)=X_val(i,:)-x_mean_train;
end
target_estimation=X_val*betahat+y_mean_train;
%vergleich=[target_estimation y_val]
NSE=(y_val-target_estimation)'*(y_val-target_estimation)/n;
end

