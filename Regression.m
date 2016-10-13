function [betahat y_mean NSE] = Regression(X,y,lambda)
    %dimensionality
    n=size(X,1);
    d=size(X,2);

    %Data preprocessing: get rid of bias parameter (beta_0)
    x_mean=mean(X,1);
    y_mean=mean(y);

    for i=1:n
        X(i,:)=X(i,:)-x_mean;
    end
    y=y-y_mean;

    %the estimate parameter
    betahat = (X'*X-eye(d)*lambda)^-1*X'*y;
    %normalized squared error
    NSE=(y-X*betahat)'*(y-X*betahat)/n;

    %plots
%     for i=1:d
%         figure(i)
%         plot(X(:,i),y,'c*');
%         hold on
%         Xi_min=min(X(:,i));
%         Xi_max=max(X(:,i));
%         xi=Xi_min:(Xi_max-Xi_min)/20:Xi_max;
%         plot(xi,xi*betahat(i))
%         j=num2str(i); 
%         title(['Training: Projection ',j]);
%         hold off
%     end
end

