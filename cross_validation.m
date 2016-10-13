function [NSE target_estimation]= cross_validation(X_val,y_val,betahat,y_mean_train)
n=size(X_val,1);
x_mean=mean(X_val,1);
for i=1:n
    X_val(i,:)=X_val(i,:)-x_mean;
end
y_mean=mean(y_val);
y_val=y_val-y_mean;
size(betahat)
NSE=(y_val-X_val*betahat)'*(y_val-X_val*betahat)/n;
target_estimation=X_val*betahat+y_mean_train;
    
%plots
%     for i=1:d
%             
%         figure(i+d)
%         plot(X_val(:,i),y_val,'c*');
%         hold on
%         Xi_min=min(X_val(:,i));
%         Xi_max=max(X_val(:,i));
%         xi=Xi_min:(Xi_max-Xi_min)/20:Xi_max;
%         plot(xi,xi*betahat(i))
%         j=num2str(i); 
%         title(['Validationset: Projection ',j]);
%         hold off
%     end


end

