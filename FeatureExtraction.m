function [X I_out] = FeatureExtraction(filename,interval,y,I_tr,N,dl)
    F4=[];
    X=[];
    for i=interval(1):interval(2) %max=278
        j=num2str(i);    
        train_j=strcat(filename,j,'.nii');   
        Data=load_nii(train_j);
        Xi=[];
        %absolut mean
        f1=mean(mean(mean(Data.img,1),2),3);
        Xi=[Xi f1];      
        % mean over neighbor boxes
        f4=mean(mean(mean(Data.img(1:1+dl,1:1+dl,1:1+dl),1),2),3);
        for k1=1+dl:dl:size(Data.img,1)-dl
            for k2=1+dl:dl:size(Data.img,2)-dl
              for k3=1+dl:dl:size(Data.img,3)-dl
                    f4=[f4 mean(mean(mean(Data.img(k1:k1+dl,k2:k2+dl,k3:k3+dl),1),2),3)];
              end
            end
        end
        F4=[F4;f4];
        X=[X;Xi];
    end
  
    %Max correlation
    size_F4=size(F4);
    if N>size(F4,2)
        N=size(F4,2);
    end
    if I_tr==0
        corr=[];
        for i=1:size(F4,2)
            size(F4,1)
            length(y)
            C=cov(F4(:,i),y);
            corr=[corr abs(C(2))];
        end   
        sorted_corr=sort(corr);
        for i=0:N-1
           [~, I]=max(sorted_corr(end-i)==corr);
            X=[X F4(:,I)];
            I_out(i+1)=I;       
        end
    else
        for i=0:N-1
            X=[X F4(:,I_tr(i+1))];
        end
        I_out=I_tr;
    end
NumberOfFeatures=size(X,2)
end

