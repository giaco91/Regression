function V = BoxToVoxel(B)
%takes a box of voxels and returns a row vector containing the amount of
%each voxels
l=size(B,1);
m=size(B,2);
n=size(B,3);
bin=10;
V=zeros(1,round(5000/bin));
for a=1:l
    for b=1:m
        for c=1:n
           V(round(B(a,b,c)/bin)+1)=V(round(B(a,b,c)/bin)+1)+1;
        end
    end
end
V=V(7:round(2000/bin));%Erstes Voxel ist immer gleich
end

