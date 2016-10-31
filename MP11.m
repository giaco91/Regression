clear all

%Hyperparameter
method='PLS';%'ridge','lasso',PLS','PCR','GPR'
submission=0;
lambda=0.2;
w=[1 1.6];%weights 1 1.6
w=w/sum(w);%control

X1=[csvread('Box88104_train') csvread('BoxE30_train')  csvread('BoxVar15_train')];
X1=zscore(X1);
X2=[csvread('VoxBox_train') csvread('Box4452_train')];
X2=zscore(X2);

y=csvread('targets.csv');

%amount of crossvalidation
if submission==1
    cv=1;
else
    cv=28;
end
for i=1:cv
valblock=10;
valbegin=round((278-valblock)/cv*i);
val_set=[valbegin valbegin+valblock];%max91 120,1 30,181 210, I=[23 119 196]
if submission==1
    train_set=[1 2 3 278];
else
    train_set=[1 val_set(1)-1 val_set(2)+1 278]; %prozentual oder zeilenvektor (max 278)
end


y_train=[y(train_set(1):train_set(2),:);y(train_set(3):train_set(4),:)];
X1_train=[X1(train_set(1):train_set(2),:);X1(train_set(3):train_set(4),:)];
X2_train=[X2(train_set(1):train_set(2),:);X2(train_set(3):train_set(4),:)];

if submission==0
    y_val=y(val_set(1):val_set(2),:);
    X1_val=X1(val_set(1):val_set(2),:);
    X2_val=X2(val_set(1):val_set(2),:);
end
%regression
[betahat1 y_mean_train x1_mean_train Loss_train1 gprMdl1]=Regression(X1_train,y_train,lambda,'PLS',9);
[betahat2 y_mean_train x2_mean_train Loss_train2 gprMdl2]=Regression(X2_train,y_train,lambda,'PLS',8);


%cross-validation
    if submission==0
        X1_val=X1_val-repmat(x1_mean_train,size(X1_val,1),1);
        X2_val=X2_val-repmat(x2_mean_train,size(X2_val,1),1);
   
        target_estimation_val=[];
        [Loss_val1 target_estimation_val(:,1)]=cross_validation(X1_val,y_val,betahat1,y_mean_train);
        [Loss_val2 target_estimation_val(:,2)]=cross_validation(X2_val,y_val,betahat2,y_mean_train);
        target_estimation_val=target_estimation_val*w';
        target_estimation_val=dataprior(target_estimation_val);
             
        error1(i)=sum((target_estimation_val-y_val).^2)/length(y_val);

    end
end
if submission==0
    meanerror1=mean(error1)
end


%write Submission file
if submission==1
    X1=[csvread('Box88104_test') csvread('BoxE30_test')  csvread('BoxVar15_test')];
    X1=zscore(X1);
    X2=[csvread('VoxBox_test') csvread('Box4452_test')];
    X2=zscore(X2);
            
        target_estimation_test(:,1)=test_validation(X1,betahat1,y_mean_train,x1_mean_train);
        target_estimation_test(:,2)=test_validation(X2,betahat2,y_mean_train,x2_mean_train);

        target_estimation_test=target_estimation_test*w';
        target_estimation_test=dataprior(target_estimation_test);

    header={'ID','Prediction'};
    submission_data=[[1:138]',target_estimation_test];
    csvwrite_with_headers('submission.csv',submission_data,header)
end
