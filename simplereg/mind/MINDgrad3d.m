function [Sx,Sy,Sz,S]=MINDgrad3d(mind1,mind2)
[m,n,o,p]=size(mind1);

A1=mean(abs(volshift(mind2,1,0,0)-mind1),4);
A2=mean(abs(volshift(mind2,-1,0,0)-mind1),4);
Sx=A1-A2;
A1=mean(abs(volshift(mind2,0,1,0)-mind1),4);
A2=mean(abs(volshift(mind2,0,-1,0)-mind1),4);
Sy=A1-A2;
A1=mean(abs(volshift(mind2,0,0,1)-mind1),4);
A2=mean(abs(volshift(mind2,0,0,-1)-mind1),4);
Sz=A1-A2;
S=mean(abs(mind2-mind1),4);

Sx(:,[1,n],:)=0;
Sy([1,m],:,:)=0;
Sz(:,:,[1,o])=0;

function vol1shift=volshift(vol1,x,y,z)

x=round(x);
y=round(y);
z=round(z);

[m,n,o,p]=size(vol1);

vol1shift=zeros(size(vol1));

x1s=max(1,x+1);
x2s=min(n,n+x);

y1s=max(1,y+1);
y2s=min(m,m+y);

z1s=max(1,z+1);
z2s=min(o,o+z);

x1=max(1,-x+1);
x2=min(n,n-x);

y1=max(1,-y+1);
y2=min(m,m-y);

z1=max(1,-z+1);
z2=min(o,o-z);

vol1shift(y1:y2,x1:x2,z1:z2,:)=vol1(y1s:y2s,x1s:x2s,z1s:z2s,:);
