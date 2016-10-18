function [betahat y_mean x_mean NSE gprMdl] = Regression(X,y,lambda,method,N)
    %dimensionality
    n=size(X,1);
    d=size(X,2);
    %Data preprocessing: get rid of bias parameter (beta_0)
    x_mean=mean(X,1);
    y_mean=mean(y);
    for i=1:n
        X(i,:)=X(i,:)-x_mean;
    end
    ym=y-y_mean;
    gprMdl=0;
    if strcmp(method,'lasso')==1
        betahat = lasso(X,ym,'Lambda',lambda);
        yhat=X*betahat(:,ceil(end/2));       
    elseif strcmp(method,'PLS')==1
        [Xl,Yl,Xs,Ys,betahatcv,PLSPctVar,PLSmsep] = plsregress(X,y,N,'CV',10);
        [NSE,I]=min(PLSmsep(2,:));
        crossvalopt=[NSE,I]     
        [Xloadings,Yloadings,Xscores,Yscores,betahat,PLSPctVar] = plsregress(X,y,I);
        betahat=betahat(2:end);
        yhat=X*betahat;
    elseif strcmp(method,'PCR')==1    
        [PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
        betaPCR = regress(ym, PCAScores(:,1:N));%Regression with the first N components 
        betahat = PCALoadings(:,1:N)*betaPCR;%transform the betas back
        yhat = X*betahat;
    elseif strcmp(method,'GPR')==1
        gprMdl = fitrgp(X,y,'KernelFunction','squaredexponential');
        yhat=predict(gprMdl,X)-y_mean;
        betahat=0;
    else
    %ridge
    betahat = (X'*X-eye(d)*lambda)^-1*X'*ym;
    yhat=X*betahat;
    end
    %normalized squared error
    NSE=(ym-yhat)'*(ym-yhat)/n;

%     plots
    if strcmp(method,'PLS')+strcmp(method,'PLS')==0 && d<10

%         for i=1:d
%             figure(i)
%             plot(X(:,i),y,'c*');
%             hold on
%             [sortedX I]=sort(X(:,i));
%             plot(sortedX,yhat(I))
%             j=num2str(i); 
%             title(['Training: Projection ',j]);
%             hold off        
%         end
    end
end

