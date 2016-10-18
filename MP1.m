clear

%Hyperparameter
method='PLS';%'ridge','lasso',PLS','PCR','GPR'
voxel=1;
train_size=[50 278]; %prozentual oder zeilenvektor (max 278)
val_size=[1 12];
lambda=0.2;
loop_over_lambda=1;
lambda_max=100;
loop_over_complexity=1;
NP=5000;%nehme so viel wie möglich
Nmin=10;
Nmax=10;
dl=10;%würfelgrösse
submission=1;%1 for writing submissionfile

if size(train_size,2)+size(val_size,2)==2
    if val_size+train_size>=1
        val_size=1-train_size;
    end
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
if submission==1
    train_set=[1 278];
elseif size(train_size,2)>1 
    train_set=train_size;
else
    train_set=[1,floor(length(y)*train_size)];%sampleinterval
end
if size(val_size,2)>1
    val_set=val_size;
else
    val_set = [train_set(2)+1,train_set(2)+1+floor(length(y)*val_size)];
end
test_set= [1 138];
y_train=y(train_set(1):train_set(2),:);
if submission==0
    y_val=y(val_set(1):val_set(2),:);
end

betahats=[];
Losses_train=[];
Losses_val=[];
for i=1:loop_over_lambda
for N=Ns
    %featureextraction
    if strcmp(method,'PLS')+strcmp(method,'PCR')==1
        M=NP;
        'PLS or PCR method'
    else
        M=N;
        'other method than PLS or PCR'
    end
    if voxel==1
        'use voxel-features'
        X=csvread('VOXEL_train');
        X_train=X(train_set(1):train_set(2),:);
        if strcmp(method,'PLS')+strcmp(method,'PCR')==0
            [X_train I_tr]=MaxCov(X_train,M,0,y_train);
        end
        if submission==0
            X_val=X(val_set(1):val_set(2),:);
            if strcmp(method,'PLS')+strcmp(method,'PCR')==0
                [X_val I_tr]=MaxCov(X_val,M,I_tr,0);
            end
        end
    else
    [X_train I_tr]=FeatureExtractionTEST('train_',train_set,y_train,0,M,dl);
    if submission==0
        [X_val bla]=FeatureExtractionTEST('train_',val_set,0,I_tr,M,dl);
    end
    end
    %regression
    [betahat y_mean_train x_mean_train Loss_train gprMdl]=Regression(X_train,y_train,lambda(i),method,N);
    if loop_over_complexity==1
        betahats=[betahats betahat];
    end
    Losses_train=[Losses_train Loss_train];
    
    %     cross-validation
    if submission==0
        if strcmp(method,'GPR')==1
            n=size(X_val,1);
            target_estimation=predict(gprMdl,X_val);
            vergleich=[target_estimation(1:10) y_val(1:10)]
            Loss_val = (y_val-target_estimation)'*(y_val-target_estimation)/n;
        else
            [Loss_val target_estimation_val]=cross_validation(X_val,y_val,betahat,y_mean_train,x_mean_train);
        end
        Losses_val=[Losses_val Loss_val];
    end
end
end

%plots
if loop_over_lambda>1 && submission==0
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
    betahat=betahats(:,I);
elseif loop_over_complexity>1 && submission==0
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
    if submission==0
        Loss_val=Losses_val
    end
end


%write Submission file
if submission==1
    header={'ID','Prediction'};
    if strcmp(method,'PLS')==1
        Nmin=NP;
    end
    if voxel==1
        X=csvread('VOXEL_test');
        X_test=X(test_set(1):test_set(2),:);
    else
        X_test=FeatureExtractionTEST('test_',test_set,0,I_tr,Nmin,dl);
    end
    if strcmp(method,'GPR')==1
        target_estimation_test=predict(gprMdl,X_test);
    else
        target_estimation_test = test_validation(X_test,betahat,y_mean_train,x_mean_train);
    end
    submission_data=[[1:test_set(2)]',target_estimation_test];
    csvwrite_with_headers('submission.csv',submission_data,header)
end



