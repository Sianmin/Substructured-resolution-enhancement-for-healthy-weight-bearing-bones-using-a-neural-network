LR_ set [패치 번호, y, x] : LR patch가 2by2로 들어가있음
HR_set [패치 번호, y, x]: HR patchrk 20by20으로 들어가있음
Dis_set [패치번호,  y, x, channel]: displacment가 들어가있음 channel 1은 하중 케이스 1의 UX channel 2는 하중 케이스 1의 UY channel 3는 하중 케이스 2의 UX ...
Subject_set[패치번호] : 해당 패치가 몇 번재 subject에 속하는지. 학습할 땐 필요없음
Position_set[패치번호] : 해당 LR 패치의 [0,0]이 Low resolution에서의 좌표값[x,y]을 나타냄. 나중에 패치들 이어붙일 때 사용됨.