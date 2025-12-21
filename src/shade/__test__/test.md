# Test 1

<p align="center">
    <img src="./imgs/test_1.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ❌        |
| Invert Image Color | ❌        |
| ~~Save ASCII~~     | ❌        |

```commandline
python shade.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1
```

---

# Test 2

<p align="center">
    <img src="./imgs/test_2.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ✅        |
| Invert Image Color | ❌        |
| ~~Save ASCII~~     | ❌        |

```commandline
python shade.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--color_option original
```

---

# Test 3

<p align="center">
    <img src="./imgs/test_3.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ✅        |
| Invert Image Color | ✅        |
| ~~Save ASCII~~     | ❌        |

```commandline
python shade.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--color_option original ^
--invert_color
```

---

# Test 4

<p align="center">
    <img src="./imgs/test_4.png" width="400">
</p>

| Test               | Property |
|--------------------|----------|
| Color Image        | ❌        |
| Invert Image Color | ✅        |
| ~~Save ASCII~~     | ❌        |

```commandline
python shade.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--invert_color
```

---

# Test 5

```text
ii._,_i;_,=  __ _:, ,  ,,;';;^!; !,! :i
+_,^,_i;^;i_^^:  ' _:-,^^,^ .;'  ii!;,,
^i^i;i^i!!^!_^^ -_---:!:_'  ;;-;_i;,!^:
;,i:ii:,,^:+!^, ,^^' MM@@@@@%^:^]+^,_, 
!,i;i;ii:;ii,;+-_%@@@@@@@@@@W[;^-i:^;! 
 ^+i;!ii;i,;^^%@@@@@MWW@@@@@$@[;-i^^^^^
:,,iiiiii;,'_@@@@@@@@@&WMW@W%@[;l^;,_. 
^ii=iiiii;^_+@@@@@@$@@@@&%+l&@M&l=i ;,.
^^iii!i+;i; @@@@@@@@MW@%%[*+[@%&+.],![;
__!iii:!i=;i@@@@@@@@%i****+[[@$@%_:!;;;
-!;ii!i!;,i^@@@@@@@$%[l[+++l=@@@@++i^!,
^^iiiii;^^!^@@@&@@@@@&[[l[+ll@@M@[_,^i_
_ii:iii;i^^,@@W%@@@@W*[[=]*i*@@%@[;! ;^
 !'^iii;! ^:@@@W@@@W%][*l* ^:W@@@B&, :,
,;;,iiii;i;;;@@@W@W%%%[M@;:*,W@@@@&; ,i
ii;,;;;;;;!!i@@@@$&&B&[W@@@@&[@@@@B&_:^
iiiiiii!i;^;;i@@%%M@@@@@@@@@W[@@@@%&;;^
!!i+;i!:,!;!!i^^^ @@@@@%[%=*+i@@@@BW%_,
!!;_!+^^^;i+!!;-,@@@@@W%[i*iii@@@@@@@i 
 ;;ii;:!i];++!! +@@@@@W%+*il**@@@@@@@& 
_:^::!i,!;i;:i:,B@@M@@@@+][[iiW@@@@@@W_
 ^^;,:!;-ii,+_;%W@@@@W$W&l]]][B@@@ii,;;
,;-;--^_;^i_[^!@W@@@W&&&&%][*%%W$l-ii!_
;,;l,i;+  *;;!@@@@@@&%%%&%[[]]][[-!^i= 
_^!;i;!*;i;l+M@@@W%&&[[B&&]][%%[[+i:_i[
+ :i!;!*:;_iWW@@@%B&&[[&&&+]]%&[[[,^^!i
ii***];!.i,_@@@@W%%&&[l&&%*=]l*][+! ,i,
```

| Test                   | Property |
|------------------------|----------|
| ~~Color Image~~        | ❌        |
| ~~Invert Image Color~~ | ❌        |
| Save ASCII             | ✅        |

```commandline
python shade.py ^
--image_path ../../resource/imgs/girl_with_pearl_earring.jpg ^
--resize_factor 1 ^
--save_ascii
```
