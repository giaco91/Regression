clear

%Hyperparameter
train_size=0.8; %Anzahl samples, max 278
val_size=0.2;
lambda=0.25;
loop_over_lambda=1;
lambda_max=5;
loop_over_complexity=1;
Nmin=60;
Nmax=75;
submission=0;%1 for writing submissionfile

if val_size+train_size>1
    val_size=1-train_size;
end
if loop_over_complexity>1
    loop_over_lambda=1;
    Ns=Nmin:(Nmax-Nmin)/loop_over_complexity:Nmax;
elseif loop_over_lambda>1
    loop_over_complexity=1;
    Ns=Nmin:Nmin;
    lambda=lambda_max;
    for i=1:loop_over_lambda-1
        lambda=[lambda lambda(end)/2];
    end
else 
    Ns=Nmin:Nmin;
end

%Read in
y=csvread('targets.csv');%size 278x1

train_set=[1,floor(length(y)*train_size)];%sampleinterval 
val_set = [train_set(2)+1 train_set(2)+1+floor(size(y)*val_size)];
test_set= [1 138];

y_train=y(train_set(1):train_set(2),:);
y_val=y(val_set(1):val_set(2),:);

betahats=[];
Losses_train=[];
Losses_val=[];
for i=1:loop_over_lambda
for N=Ns
    %featureextraction
    [X_train I_tr1]=FeatureExtraction('train_',train_set,y_train,0,N);
    [X_val bla]=FeatureExtraction('train_',val_set,0,I_tr1,N);
    
    %regression
    [betahat y_mean_train Loss_train]=Regression(X_train,y_train,lambda(i));
    if loop_over_complexity==1
        betahats=[betahats betahat];
    end
    Losses_train=[Losses_train Loss_train];
    
    %cross-validation
    [Loss_val target_estimation_val]=cross_validation(X_val,y_val,betahat,y_mean_train);
    Losses_val=[Losses_val Loss_val];
end
end

%plots
if loop_over_lambda>1
    figure(10^7);
    plot(lambda,Losses_train);
    hold on
    plot(lambda,Losses_val);
    size_beta=diag(betahats'*betahats);
    plot(lambda,size_beta/max(size_beta)*max(Losses_val));
    [MIN,I]=min(Losses_val);
    title(['Regularization Curve mit optimalem Lambda=',num2str(lambda(I))]);
    xlabel('lambda');
    ylabel('Loss/Param');
    legend('train','val','size beta');
    hold off
    
    Loss_train_opt=Losses_train(I)
    Loss_val_opt=Losses_val(I)
    betahat=betahats(:,I)
elseif loop_over_complexity>1
    figure(10^7);
    plot(Ns,Losses_train);
    hold on
    plot(Ns,Losses_val);
    [MIN,I]=min(Losses_val);
    title(['Complexity Curve with optimal N=',num2str(Ns(I))]);
    xlabel('N');
    ylabel('Loss');
    legend('train','val');
    hold off
    
    Loss_train_opt=Losses_train(I)
    Loss_val_opt=Losses_val(I)
else
    Loss_train=Losses_train
    Loss_val=Losses_val
    lambda=lambda
end


%write Submission file
if submission==1
    header={'ID','Prediction'};
    X_test=FeatureExtraction('test_',test_set,0,I_tr1, I_tr2,Nmin);
    target_estimation_test = test_validation(X_test,betahat,y_mean_train);
    submission_data=[[1:test_set(2)]',target_estimation_test];
    csvwrite_with_headers('submission.csv',submission_data,header)
end



