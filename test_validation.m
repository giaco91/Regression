function target_estimation = test_validation(X_test,betahat,y_mean_train)
n=size(X_test,1);
x_mean=mean(X_test,1);
for i=1:n
    X_test(i,:)=X_test(i,:)-x_mean;
end
target_estimation=X_test*betahat+y_mean_train;
end

