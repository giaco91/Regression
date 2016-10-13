function [X I_out1] = FeatureExtraction(filename,interval,y,I_tr1,N)
    F4=[];
    for i=interval(1):interval(2) %max=278
        j=num2str(i);    
        train_j=strcat(filename,j,'.nii');   
        Data=load_nii(train_j);
        Xi=[];
        %absolut mean
        f1=mean(mean(mean(Data.img,1),2),3);
        Xi=[Xi f1];      
        % mean over neighbor boxes
        dl=40;
        f4=mean(mean(mean(Data.img(1:1+dl,1:1+dl,1:1+dl),1),2),3);
        for k1=1+dl:dl:size(Data.img,1)-dl
            for k2=1+dl:dl:size(Data.img,2)-dl
              for k3=1+dl:dl:size(Data.img,3)-dl
                    f4=[f4 mean(mean(mean(Data.img(k1:k1+dl,k2:k2+dl,k3:k3+dl),1),2),3)];
                end
            end
        end
        F4=[F4;f4];

        %save in Feature-Sample matrix
        if i==interval(1)
            X=Xi;
        else
            X=[X;Xi];
        end
    end
    
    %Max correlation
    size_F4=size(F4);
    if N>size(F4,2)
        N=size(F4,2);
    end
    if I_tr1==0
        corr1=[];
        for i=1:size(F4,2)
            C1=cov(F4(:,i),y);
            corr1=[corr1 C1(2)];
        end   
        sorted_corr1=sort(corr1);
        for i=0:N-1
           [~, I1]=max(sorted_corr1(end-i)==corr1);
           [~, I11]=max(sorted_corr1(i+1)==corr1);
            X=[X F4(:,I1).*F4(:,I11)];
            I_out1(i+1)=I1;       
        end
    else
        for i=0:N-1
            X=[X F4(:,I_tr1(i+1))];
        end
        I_out1=I_tr1;
    end
NumberOfFeatures=size(X,2)
end

