clear

%Hyperparameter
train_size=0.9; %Anzahl samples, max 278
val_size=0.1;
N=50;

if val_size+train_size>1
    val_size=1-train_size;
end


%Read in
y=csvread('targets.csv');%size 278x1

train_set=[1,floor(length(y)*train_size)];%sampleinterval 
val_set = [train_set(2)+1 train_set(2)+1+floor(size(y)*val_size)];

y_train=y(train_set(1):train_set(2),:);
y_val=y(val_set(1):val_set(2),:);

[X_train I_tr1 I_tr2]=FeatureExtraction('train_',train_set,y_train,0,0,N);
[X_val bla bla]=FeatureExtraction('train_',val_set,0,I_tr1, I_tr2,N);

%feedforwardnet
net = feedforwardnet(10);
net = train(net,X_train',y_train');
y_val_hat = net(X_val')';
n=size(X_val,1);
NSE=(y_val-y_val_hat)'*(y_val-y_val_hat)/n

%regression
% b = regress(y_train,X_train);
% n=size(X_val,1);
% NSE=(y_val-X_val*b)'*(y_val-X_val*b)/n
