??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8??
?
batch_normalization/gammaVarHandleOp**
shared_namebatch_normalization/gamma*
shape:*
dtype0*
_output_shapes
: 
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*)
shared_namebatch_normalization/beta*
dtype0*
shape:*
_output_shapes
: 
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *0
shared_name!batch_normalization/moving_mean*
dtype0*
shape:
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*4
shared_name%#batch_normalization/moving_variance*
_output_shapes
: *
shape:*
dtype0
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
dtype0*
shared_namedense_2/kernel*
_output_shapes
: *
shape
:@
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
shape:@*
dtype0*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:@
t
dense/kernelVarHandleOp*
_output_shapes
: *
shared_namedense/kernel*
shape
:*
dtype0
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
shape:*
shared_name
dense/bias*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
?
batch_normalization_2/gammaVarHandleOp*,
shared_namebatch_normalization_2/gamma*
_output_shapes
: *
dtype0*
shape:@
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance*
dtype0*
shape:@
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:@
x
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*
dtype0*
shape
:*
_output_shapes
: 
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:
p
dense_1/biasVarHandleOp*
_output_shapes
: *
shared_namedense_1/bias*
dtype0*
shape:
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
x
dense_3/kernelVarHandleOp*
shared_namedense_3/kernel*
_output_shapes
: *
dtype0*
shape
:@@
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:@@
p
dense_3/biasVarHandleOp*
dtype0*
shared_namedense_3/bias*
shape:@*
_output_shapes
: 
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:@
?
batch_normalization_3/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes
:@
?
batch_normalization_3/betaVarHandleOp*
dtype0*+
shared_namebatch_normalization_3/beta*
_output_shapes
: *
shape:@
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*2
shared_name#!batch_normalization_3/moving_mean*
_output_shapes
: *
shape:@*
dtype0
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:@
?
%batch_normalization_3/moving_varianceVarHandleOp*
shape:@*6
shared_name'%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:@
?
batch_normalization_1/gammaVarHandleOp*,
shared_namebatch_normalization_1/gamma*
_output_shapes
: *
dtype0*
shape:
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *+
shared_namebatch_normalization_1/beta*
shape:*
dtype0
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean*
_output_shapes
: 
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
%batch_normalization_1/moving_varianceVarHandleOp*
dtype0*6
shared_name'%batch_normalization_1/moving_variance*
shape:*
_output_shapes
: 
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
x
dense_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:T*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:T*
dtype0
p
dense_4/biasVarHandleOp*
shape:*
dtype0*
shared_namedense_4/bias*
_output_shapes
: 
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes
:
|
training/Adamax/iterVarHandleOp*
shape: *
dtype0	*%
shared_nametraining/Adamax/iter*
_output_shapes
: 
u
(training/Adamax/iter/Read/ReadVariableOpReadVariableOptraining/Adamax/iter*
dtype0	*
_output_shapes
: 
?
training/Adamax/beta_1VarHandleOp*
shape: *'
shared_nametraining/Adamax/beta_1*
_output_shapes
: *
dtype0
y
*training/Adamax/beta_1/Read/ReadVariableOpReadVariableOptraining/Adamax/beta_1*
_output_shapes
: *
dtype0
?
training/Adamax/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *'
shared_nametraining/Adamax/beta_2
y
*training/Adamax/beta_2/Read/ReadVariableOpReadVariableOptraining/Adamax/beta_2*
dtype0*
_output_shapes
: 
~
training/Adamax/decayVarHandleOp*
dtype0*
shape: *&
shared_nametraining/Adamax/decay*
_output_shapes
: 
w
)training/Adamax/decay/Read/ReadVariableOpReadVariableOptraining/Adamax/decay*
dtype0*
_output_shapes
: 
?
training/Adamax/learning_rateVarHandleOp*.
shared_nametraining/Adamax/learning_rate*
shape: *
_output_shapes
: *
dtype0
?
1training/Adamax/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adamax/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shared_nametotal*
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
shape: *
shared_namecount*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
+training/Adamax/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
shape:*
dtype0*<
shared_name-+training/Adamax/batch_normalization/gamma/m
?
?training/Adamax/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp+training/Adamax/batch_normalization/gamma/m*
dtype0*
_output_shapes
:
?
*training/Adamax/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*;
shared_name,*training/Adamax/batch_normalization/beta/m*
shape:
?
>training/Adamax/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp*training/Adamax/batch_normalization/beta/m*
_output_shapes
:*
dtype0
?
 training/Adamax/dense_2/kernel/mVarHandleOp*
dtype0*
shape
:@*
_output_shapes
: *1
shared_name" training/Adamax/dense_2/kernel/m
?
4training/Adamax/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_2/kernel/m*
dtype0*
_output_shapes

:@
?
training/Adamax/dense_2/bias/mVarHandleOp*/
shared_name training/Adamax/dense_2/bias/m*
dtype0*
_output_shapes
: *
shape:@
?
2training/Adamax/dense_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_2/bias/m*
_output_shapes
:@*
dtype0
?
training/Adamax/dense/kernel/mVarHandleOp*
_output_shapes
: */
shared_name training/Adamax/dense/kernel/m*
shape
:*
dtype0
?
2training/Adamax/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/kernel/m*
dtype0*
_output_shapes

:
?
training/Adamax/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adamax/dense/bias/m*
shape:
?
0training/Adamax/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/bias/m*
dtype0*
_output_shapes
:
?
-training/Adamax/batch_normalization_2/gamma/mVarHandleOp*
shape:@*
_output_shapes
: *>
shared_name/-training/Adamax/batch_normalization_2/gamma/m*
dtype0
?
Atraining/Adamax/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp-training/Adamax/batch_normalization_2/gamma/m*
dtype0*
_output_shapes
:@
?
,training/Adamax/batch_normalization_2/beta/mVarHandleOp*
shape:@*
_output_shapes
: *
dtype0*=
shared_name.,training/Adamax/batch_normalization_2/beta/m
?
@training/Adamax/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp,training/Adamax/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
?
 training/Adamax/dense_1/kernel/mVarHandleOp*
dtype0*1
shared_name" training/Adamax/dense_1/kernel/m*
_output_shapes
: *
shape
:
?
4training/Adamax/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
training/Adamax/dense_1/bias/mVarHandleOp*/
shared_name training/Adamax/dense_1/bias/m*
_output_shapes
: *
shape:*
dtype0
?
2training/Adamax/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_1/bias/m*
dtype0*
_output_shapes
:
?
 training/Adamax/dense_3/kernel/mVarHandleOp*
dtype0*1
shared_name" training/Adamax/dense_3/kernel/m*
shape
:@@*
_output_shapes
: 
?
4training/Adamax/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_3/kernel/m*
_output_shapes

:@@*
dtype0
?
training/Adamax/dense_3/bias/mVarHandleOp*
dtype0*/
shared_name training/Adamax/dense_3/bias/m*
_output_shapes
: *
shape:@
?
2training/Adamax/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_3/bias/m*
_output_shapes
:@*
dtype0
?
-training/Adamax/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
shape:@*>
shared_name/-training/Adamax/batch_normalization_3/gamma/m*
dtype0
?
Atraining/Adamax/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp-training/Adamax/batch_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
?
,training/Adamax/batch_normalization_3/beta/mVarHandleOp*=
shared_name.,training/Adamax/batch_normalization_3/beta/m*
dtype0*
_output_shapes
: *
shape:@
?
@training/Adamax/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp,training/Adamax/batch_normalization_3/beta/m*
dtype0*
_output_shapes
:@
?
-training/Adamax/batch_normalization_1/gamma/mVarHandleOp*>
shared_name/-training/Adamax/batch_normalization_1/gamma/m*
dtype0*
shape:*
_output_shapes
: 
?
Atraining/Adamax/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp-training/Adamax/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
,training/Adamax/batch_normalization_1/beta/mVarHandleOp*
shape:*=
shared_name.,training/Adamax/batch_normalization_1/beta/m*
dtype0*
_output_shapes
: 
?
@training/Adamax/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp,training/Adamax/batch_normalization_1/beta/m*
dtype0*
_output_shapes
:
?
 training/Adamax/dense_4/kernel/mVarHandleOp*
_output_shapes
: *1
shared_name" training/Adamax/dense_4/kernel/m*
dtype0*
shape
:T
?
4training/Adamax/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_4/kernel/m*
_output_shapes

:T*
dtype0
?
training/Adamax/dense_4/bias/mVarHandleOp*/
shared_name training/Adamax/dense_4/bias/m*
shape:*
_output_shapes
: *
dtype0
?
2training/Adamax/dense_4/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_4/bias/m*
_output_shapes
:*
dtype0
?
+training/Adamax/batch_normalization/gamma/vVarHandleOp*<
shared_name-+training/Adamax/batch_normalization/gamma/v*
shape:*
_output_shapes
: *
dtype0
?
?training/Adamax/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp+training/Adamax/batch_normalization/gamma/v*
dtype0*
_output_shapes
:
?
*training/Adamax/batch_normalization/beta/vVarHandleOp*;
shared_name,*training/Adamax/batch_normalization/beta/v*
_output_shapes
: *
dtype0*
shape:
?
>training/Adamax/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp*training/Adamax/batch_normalization/beta/v*
dtype0*
_output_shapes
:
?
 training/Adamax/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" training/Adamax/dense_2/kernel/v
?
4training/Adamax/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_2/kernel/v*
_output_shapes

:@*
dtype0
?
training/Adamax/dense_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: */
shared_name training/Adamax/dense_2/bias/v*
shape:@
?
2training/Adamax/dense_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_2/bias/v*
_output_shapes
:@*
dtype0
?
training/Adamax/dense/kernel/vVarHandleOp*
shape
:*
dtype0*/
shared_name training/Adamax/dense/kernel/v*
_output_shapes
: 
?
2training/Adamax/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/kernel/v*
dtype0*
_output_shapes

:
?
training/Adamax/dense/bias/vVarHandleOp*-
shared_nametraining/Adamax/dense/bias/v*
dtype0*
shape:*
_output_shapes
: 
?
0training/Adamax/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/bias/v*
_output_shapes
:*
dtype0
?
-training/Adamax/batch_normalization_2/gamma/vVarHandleOp*
shape:@*
_output_shapes
: *
dtype0*>
shared_name/-training/Adamax/batch_normalization_2/gamma/v
?
Atraining/Adamax/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp-training/Adamax/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
?
,training/Adamax/batch_normalization_2/beta/vVarHandleOp*=
shared_name.,training/Adamax/batch_normalization_2/beta/v*
dtype0*
shape:@*
_output_shapes
: 
?
@training/Adamax/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp,training/Adamax/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
?
 training/Adamax/dense_1/kernel/vVarHandleOp*
_output_shapes
: *1
shared_name" training/Adamax/dense_1/kernel/v*
dtype0*
shape
:
?
4training/Adamax/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_1/kernel/v*
dtype0*
_output_shapes

:
?
training/Adamax/dense_1/bias/vVarHandleOp*
shape:*
dtype0*/
shared_name training/Adamax/dense_1/bias/v*
_output_shapes
: 
?
2training/Adamax/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_1/bias/v*
dtype0*
_output_shapes
:
?
 training/Adamax/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
shape
:@@*
dtype0*1
shared_name" training/Adamax/dense_3/kernel/v
?
4training/Adamax/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_3/kernel/v*
_output_shapes

:@@*
dtype0
?
training/Adamax/dense_3/bias/vVarHandleOp*
dtype0*/
shared_name training/Adamax/dense_3/bias/v*
_output_shapes
: *
shape:@
?
2training/Adamax/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_3/bias/v*
dtype0*
_output_shapes
:@
?
-training/Adamax/batch_normalization_3/gamma/vVarHandleOp*>
shared_name/-training/Adamax/batch_normalization_3/gamma/v*
_output_shapes
: *
shape:@*
dtype0
?
Atraining/Adamax/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp-training/Adamax/batch_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
?
,training/Adamax/batch_normalization_3/beta/vVarHandleOp*=
shared_name.,training/Adamax/batch_normalization_3/beta/v*
dtype0*
_output_shapes
: *
shape:@
?
@training/Adamax/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp,training/Adamax/batch_normalization_3/beta/v*
_output_shapes
:@*
dtype0
?
-training/Adamax/batch_normalization_1/gamma/vVarHandleOp*
dtype0*>
shared_name/-training/Adamax/batch_normalization_1/gamma/v*
_output_shapes
: *
shape:
?
Atraining/Adamax/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp-training/Adamax/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
,training/Adamax/batch_normalization_1/beta/vVarHandleOp*
shape:*
_output_shapes
: *
dtype0*=
shared_name.,training/Adamax/batch_normalization_1/beta/v
?
@training/Adamax/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp,training/Adamax/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
 training/Adamax/dense_4/kernel/vVarHandleOp*
dtype0*1
shared_name" training/Adamax/dense_4/kernel/v*
_output_shapes
: *
shape
:T
?
4training/Adamax/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_4/kernel/v*
_output_shapes

:T*
dtype0
?
training/Adamax/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*/
shared_name training/Adamax/dense_4/bias/v*
shape:
?
2training/Adamax/dense_4/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?{
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *?{
value?zB?z B?z
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
?
axis
	gamma
beta
moving_mean
 moving_variance
!trainable_variables
"	variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
a
@	constants
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
a
I	constants
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
a
N	constants
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
a
Y	constants
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
?
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
?
gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
R
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
R
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
h

xkernel
ybias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
?
~iter

beta_1
?beta_2

?decay
?learning_ratem?m?%m?&m?+m?,m?2m?3m?:m?;m?Sm?Tm?_m?`m?hm?im?xm?ym?v?v?%v?&v?+v?,v?2v?3v?:v?;v?Sv?Tv?_v?`v?hv?iv?xv?yv?
?
0
1
%2
&3
+4
,5
26
37
:8
;9
S10
T11
_12
`13
h14
i15
x16
y17
?
0
1
2
 3
%4
&5
+6
,7
28
39
410
511
:12
;13
S14
T15
_16
`17
a18
b19
h20
i21
j22
k23
x24
y25
 
?
 ?layer_regularization_losses
trainable_variables
	variables
?metrics
?non_trainable_variables
?layers
regularization_losses
 
 
 
 
?
 ?layer_regularization_losses
trainable_variables
	variables
regularization_losses
?non_trainable_variables
?layers
?metrics
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
2
 3
 
?
 ?layer_regularization_losses
!trainable_variables
"	variables
#regularization_losses
?non_trainable_variables
?layers
?metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
 ?layer_regularization_losses
'trainable_variables
(	variables
)regularization_losses
?non_trainable_variables
?layers
?metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?
 ?layer_regularization_losses
-trainable_variables
.	variables
/regularization_losses
?non_trainable_variables
?layers
?metrics
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
42
53
 
?
 ?layer_regularization_losses
6trainable_variables
7	variables
8regularization_losses
?non_trainable_variables
?layers
?metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
 ?layer_regularization_losses
<trainable_variables
=	variables
>regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
 
?
 ?layer_regularization_losses
Atrainable_variables
B	variables
Cregularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
 ?layer_regularization_losses
Etrainable_variables
F	variables
Gregularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
 
?
 ?layer_regularization_losses
Jtrainable_variables
K	variables
Lregularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
 
?
 ?layer_regularization_losses
Otrainable_variables
P	variables
Qregularization_losses
?non_trainable_variables
?layers
?metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
 ?layer_regularization_losses
Utrainable_variables
V	variables
Wregularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
 
?
 ?layer_regularization_losses
Ztrainable_variables
[	variables
\regularization_losses
?non_trainable_variables
?layers
?metrics
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
a2
b3
 
?
 ?layer_regularization_losses
ctrainable_variables
d	variables
eregularization_losses
?non_trainable_variables
?layers
?metrics
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
j2
k3
 
?
 ?layer_regularization_losses
ltrainable_variables
m	variables
nregularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
 ?layer_regularization_losses
ptrainable_variables
q	variables
rregularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
 ?layer_regularization_losses
ttrainable_variables
u	variables
vregularization_losses
?non_trainable_variables
?layers
?metrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
 
?
 ?layer_regularization_losses
ztrainable_variables
{	variables
|regularization_losses
?non_trainable_variables
?layers
?metrics
SQ
VARIABLE_VALUEtraining/Adamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining/Adamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining/Adamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining/Adamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

?0
8
0
 1
42
53
a4
b5
j6
k7
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 

0
 1
 
 
 
 
 
 
 
 
 
 
 

40
51
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

a0
b1
 
 
 

j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
 
?
 ?layer_regularization_losses
?trainable_variables
?	variables
?regularization_losses
?non_trainable_variables
?layers
?metrics
 

?0
?1
 
 
??
VARIABLE_VALUE+training/Adamax/batch_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adamax/batch_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training/Adamax/batch_normalization_2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,training/Adamax/batch_normalization_2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training/Adamax/batch_normalization_3/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,training/Adamax/batch_normalization_3/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training/Adamax/batch_normalization_1/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,training/Adamax/batch_normalization_1/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+training/Adamax/batch_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*training/Adamax/batch_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training/Adamax/batch_normalization_2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,training/Adamax/batch_normalization_2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training/Adamax/batch_normalization_3/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,training/Adamax/batch_normalization_3/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-training/Adamax/batch_normalization_1/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,training/Adamax/batch_normalization_1/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adamax/dense_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adamax/dense_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
shape:?????????*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_2/kerneldense_2/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense/kernel
dense/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_4/kerneldense_4/bias*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-17337*,
f'R%
#__inference_signature_wrapper_15986*&
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp(training/Adamax/iter/Read/ReadVariableOp*training/Adamax/beta_1/Read/ReadVariableOp*training/Adamax/beta_2/Read/ReadVariableOp)training/Adamax/decay/Read/ReadVariableOp1training/Adamax/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp?training/Adamax/batch_normalization/gamma/m/Read/ReadVariableOp>training/Adamax/batch_normalization/beta/m/Read/ReadVariableOp4training/Adamax/dense_2/kernel/m/Read/ReadVariableOp2training/Adamax/dense_2/bias/m/Read/ReadVariableOp2training/Adamax/dense/kernel/m/Read/ReadVariableOp0training/Adamax/dense/bias/m/Read/ReadVariableOpAtraining/Adamax/batch_normalization_2/gamma/m/Read/ReadVariableOp@training/Adamax/batch_normalization_2/beta/m/Read/ReadVariableOp4training/Adamax/dense_1/kernel/m/Read/ReadVariableOp2training/Adamax/dense_1/bias/m/Read/ReadVariableOp4training/Adamax/dense_3/kernel/m/Read/ReadVariableOp2training/Adamax/dense_3/bias/m/Read/ReadVariableOpAtraining/Adamax/batch_normalization_3/gamma/m/Read/ReadVariableOp@training/Adamax/batch_normalization_3/beta/m/Read/ReadVariableOpAtraining/Adamax/batch_normalization_1/gamma/m/Read/ReadVariableOp@training/Adamax/batch_normalization_1/beta/m/Read/ReadVariableOp4training/Adamax/dense_4/kernel/m/Read/ReadVariableOp2training/Adamax/dense_4/bias/m/Read/ReadVariableOp?training/Adamax/batch_normalization/gamma/v/Read/ReadVariableOp>training/Adamax/batch_normalization/beta/v/Read/ReadVariableOp4training/Adamax/dense_2/kernel/v/Read/ReadVariableOp2training/Adamax/dense_2/bias/v/Read/ReadVariableOp2training/Adamax/dense/kernel/v/Read/ReadVariableOp0training/Adamax/dense/bias/v/Read/ReadVariableOpAtraining/Adamax/batch_normalization_2/gamma/v/Read/ReadVariableOp@training/Adamax/batch_normalization_2/beta/v/Read/ReadVariableOp4training/Adamax/dense_1/kernel/v/Read/ReadVariableOp2training/Adamax/dense_1/bias/v/Read/ReadVariableOp4training/Adamax/dense_3/kernel/v/Read/ReadVariableOp2training/Adamax/dense_3/bias/v/Read/ReadVariableOpAtraining/Adamax/batch_normalization_3/gamma/v/Read/ReadVariableOp@training/Adamax/batch_normalization_3/beta/v/Read/ReadVariableOpAtraining/Adamax/batch_normalization_1/gamma/v/Read/ReadVariableOp@training/Adamax/batch_normalization_1/beta/v/Read/ReadVariableOp4training/Adamax/dense_4/kernel/v/Read/ReadVariableOp2training/Adamax/dense_4/bias/v/Read/ReadVariableOpConst*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *'
f"R 
__inference__traced_save_17427*,
_gradient_op_typePartitionedCall-17428*R
TinK
I2G	
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_2/kerneldense_2/biasdense/kernel
dense/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_1/kerneldense_1/biasdense_3/kerneldense_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_4/kerneldense_4/biastraining/Adamax/itertraining/Adamax/beta_1training/Adamax/beta_2training/Adamax/decaytraining/Adamax/learning_ratetotalcount+training/Adamax/batch_normalization/gamma/m*training/Adamax/batch_normalization/beta/m training/Adamax/dense_2/kernel/mtraining/Adamax/dense_2/bias/mtraining/Adamax/dense/kernel/mtraining/Adamax/dense/bias/m-training/Adamax/batch_normalization_2/gamma/m,training/Adamax/batch_normalization_2/beta/m training/Adamax/dense_1/kernel/mtraining/Adamax/dense_1/bias/m training/Adamax/dense_3/kernel/mtraining/Adamax/dense_3/bias/m-training/Adamax/batch_normalization_3/gamma/m,training/Adamax/batch_normalization_3/beta/m-training/Adamax/batch_normalization_1/gamma/m,training/Adamax/batch_normalization_1/beta/m training/Adamax/dense_4/kernel/mtraining/Adamax/dense_4/bias/m+training/Adamax/batch_normalization/gamma/v*training/Adamax/batch_normalization/beta/v training/Adamax/dense_2/kernel/vtraining/Adamax/dense_2/bias/vtraining/Adamax/dense/kernel/vtraining/Adamax/dense/bias/v-training/Adamax/batch_normalization_2/gamma/v,training/Adamax/batch_normalization_2/beta/v training/Adamax/dense_1/kernel/vtraining/Adamax/dense_1/bias/v training/Adamax/dense_3/kernel/vtraining/Adamax/dense_3/bias/v-training/Adamax/batch_normalization_3/gamma/v,training/Adamax/batch_normalization_3/beta/v-training/Adamax/batch_normalization_1/gamma/v,training/Adamax/batch_normalization_1/beta/v training/Adamax/dense_4/kernel/vtraining/Adamax/dense_4/bias/v*Q
TinJ
H2F*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*
Tout
2**
f%R#
!__inference__traced_restore_17647*,
_gradient_op_typePartitionedCall-17648??
?
\
@__inference_re_lu_layer_call_and_return_conditional_losses_16780

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
x
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_16791
inputs_0
inputs_1
identityR
mul_1Mulinputs_0inputs_1*'
_output_shapes
:?????????*
T0Q
IdentityIdentity	mul_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_16835

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 dense_3/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
5__inference_batch_normalization_1_layer_call_fn_17085

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14986*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14985*'
_output_shapes
:?????????*
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
5__inference_batch_normalization_2_layer_call_fn_16730

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14712*
Tout
2*'
_output_shapes
:?????????@*,
_gradient_op_typePartitionedCall-14713*
Tin	
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
%__inference_dense_layer_call_fn_16610

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-15137*
Tin
2*'
_output_shapes
:?????????*
Tout
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15131**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
__inference_loss_fn_2_17178=
9dense_1_kernel_regularizer_square_readvariableop_resource
identity??0dense_1/kernel/Regularizer/Square/ReadVariableOp?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
valueB"       *
_output_shapes
:?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentity"dense_1/kernel/Regularizer/add:z:01^dense_1/kernel/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:  
?7
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17053

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
_output_shapes

:*
T0?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*
_output_shapes

:m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
_output_shapes
:*
T0*
squeeze_dims
 ?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*
_output_shapes
: *
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*
dtype0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 T
batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes
:*
T0?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????*
T0h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
?
?
%__inference_model_layer_call_fn_15765
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-15736*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15735?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:
 : : : : : : : : : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : : : : :	 
?
?
'__inference_dense_3_layer_call_fn_16842

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15239*,
_gradient_op_typePartitionedCall-15245*'
_output_shapes
:?????????@*
Tin
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_15131

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
?#<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
'__inference_dense_2_layer_call_fn_16577

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15073*,
_gradient_op_typePartitionedCall-15079*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?7
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16689

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
_output_shapes
: *
dtype0
N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*
_output_shapes

:@d
moments/StopGradientStopGradientmoments/mean:output:0*
_output_shapes

:@*
T0?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*'
_output_shapes
:?????????@*
T0l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
	keep_dims(*
_output_shapes

:@m
moments/SqueezeSqueezemoments/mean:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0s
moments/Squeeze_1Squeezemoments/variance:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
valueB
 *
?#<*
dtype0?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:@?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *
?#<*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????@*
T0h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
_output_shapes
:@*
T0?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
?
[
/__inference_tf_op_layer_add_layer_call_fn_16809
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*,
_gradient_op_typePartitionedCall-15289*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_15282*'
_output_shapes
:?????????`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?|
?
@__inference_model_layer_call_and_return_conditional_losses_15623

inputs6
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_48
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14524*
Tin	
2*
Tout
2*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14523?
dense_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*'
_output_shapes
:?????????@*
Tin
2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15073**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15079*
Tout
2?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14677**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-14678*'
_output_shapes
:?????????@*
Tin	
2?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15137*
Tout
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15131*
Tin
2**
config_proto

GPU 

CPU2J 8?
re_lu/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_15153*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????@*,
_gradient_op_typePartitionedCall-15159*
Tin
2?
tf_op_layer_mul/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense/StatefulPartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-15179*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_15172*
Tout
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15210*
Tout
2*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15204**
config_proto

GPU 

CPU2J 8?
dense_3/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15239*,
_gradient_op_typePartitionedCall-15245*
Tout
2*'
_output_shapes
:?????????@*
Tin
2?
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-15269*'
_output_shapes
:?????????*U
fPRN
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_15262*
Tout
2**
config_proto

GPU 

CPU2J 8?
tf_op_layer_add/PartitionedCallPartitionedCall(tf_op_layer_mul/PartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_15282*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15289*
Tin
2?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14832*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14831*
Tin	
2*
Tout
2*'
_output_shapes
:?????????@?
!tf_op_layer_add_1/PartitionedCallPartitionedCall*tf_op_layer_mul_1/PartitionedCall:output:0(tf_op_layer_add/PartitionedCall:output:0*'
_output_shapes
:?????????*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15332*
Tout
2*U
fPRN
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_15325?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*tf_op_layer_add_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14986*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14985*'
_output_shapes
:?????????*
Tin	
2*
Tout
2?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-15373*
Tin
2*'
_output_shapes
:?????????@*K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_15367**
config_proto

GPU 

CPU2J 8*
Tout
2?
concatenate/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*'
_output_shapes
:?????????T**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_15387*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCall-15394?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15412*'
_output_shapes
:?????????*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-15418?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_statefulpartitionedcall_args_1 ^dense_2/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@*
T0q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_statefulpartitionedcall_args_1^dense/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:o
dense/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_statefulpartitionedcall_args_1 ^dense_1/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_statefulpartitionedcall_args_1 ^dense_3/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@@*
T0q
 dense_3/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?7
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14985

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
_output_shapes
: *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*
_output_shapes

:d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
squeeze_dims
 *
_output_shapes
:s
moments/Squeeze_1Squeezemoments/variance:output:0*
squeeze_dims
 *
_output_shapes
:*
T0?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:*
T0?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 *
dtype0?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:*
T0?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 T
batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
_output_shapes
:*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
?
\
@__inference_re_lu_layer_call_and_return_conditional_losses_15153

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????@Z
IdentityIdentityRelu:activations:0*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?	
?
B__inference_dense_4_layer_call_and_return_conditional_losses_15412

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Ti
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????T::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
]
1__inference_tf_op_layer_add_1_layer_call_fn_16854
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*U
fPRN
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_15325**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15332*
Tout
2*
Tin
2*'
_output_shapes
:?????????`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14712

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
_output_shapes
:@*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes
:@*
T0?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:@*
T0c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes
:@*
T0?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:@*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????@*
T0?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2: :& "
 
_user_specified_nameinputs: : : 
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16712

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@T
batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
_output_shapes
:@*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:@*
T0c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????@*
T0?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes
:@*
T0?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:@*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
?
t
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_15282

inputs
inputs_1
identityP
addAddV2inputsinputs_1*'
_output_shapes
:?????????*
T0O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?7
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16933

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 ZN
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
_output_shapes

:@*
T0*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
_output_shapes

:@*
T0?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*'
_output_shapes
:?????????@*
T0l
"moments/variance/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
_output_shapes

:@*
	keep_dims(*
T0m
moments/SqueezeSqueezemoments/mean:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 ?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *
?#<*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:@?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 ?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0*
_output_shapes
:@?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
_output_shapes
:@*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes
:@*
T0?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????@*
T0h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
_output_shapes
:@*
T0?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
_output_shapes
:@*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????@*
T0?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
??
?
@__inference_model_layer_call_and_return_conditional_losses_16362

inputs9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?0batch_normalization_1/batchnorm/ReadVariableOp_1?0batch_normalization_1/batchnorm/ReadVariableOp_2?2batch_normalization_1/batchnorm/mul/ReadVariableOp?.batch_normalization_2/batchnorm/ReadVariableOp?0batch_normalization_2/batchnorm/ReadVariableOp_1?0batch_normalization_2/batchnorm/ReadVariableOp_2?2batch_normalization_2/batchnorm/mul/ReadVariableOp?.batch_normalization_3/batchnorm/ReadVariableOp?0batch_normalization_3/batchnorm/ReadVariableOp_1?0batch_normalization_3/batchnorm/ReadVariableOp_2?2batch_normalization_3/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOpb
 batch_normalization/LogicalAnd/xConst*
value	B
 Z *
_output_shapes
: *
dtype0
b
 batch_normalization/LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ?
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: ?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
_output_shapes
:*
T0?
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_2/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0d
"batch_normalization_2/LogicalAnd/xConst*
dtype0
*
value	B
 Z *
_output_shapes
: d
"batch_normalization_2/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ?
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: ?
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@j
%batch_normalization_2/batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: ?
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:@*
T0?
%batch_normalization_2/batchnorm/mul_1Muldense_2/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
_output_shapes
:@*
T0?
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
_output_shapes
:@*
T0?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*'
_output_shapes
:?????????@*
T0?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????o

re_lu/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
T0?
tf_op_layer_mul/mulMul'batch_normalization/batchnorm/add_1:z:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
dense_1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
dense_3/MatMulMatMulre_lu/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
tf_op_layer_mul_1/mul_1Mul'batch_normalization/batchnorm/add_1:z:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
tf_op_layer_add/addAddV2tf_op_layer_mul/mul:z:0'batch_normalization/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
T0d
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
d
"batch_normalization_3/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z?
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: ?
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
_output_shapes
:@*
T0|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:@*
T0?
%batch_normalization_3/batchnorm/mul_1Muldense_3/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
_output_shapes
:@*
T0?
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
_output_shapes
:@*
T0?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
tf_op_layer_add_1/add_1AddV2tf_op_layer_mul_1/mul_1:z:0tf_op_layer_add/add:z:0*'
_output_shapes
:?????????*
T0d
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z d
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: ?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o?:*
dtype0?
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
_output_shapes
:*
T0|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
_output_shapes
:*
T0?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/mul_1Multf_op_layer_add_1/add_1:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
_output_shapes
:*
T0?
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
_output_shapes
:*
T0?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????q
re_lu_1/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
T0Y
concatenate/concat/axisConst*
value	B :*
_output_shapes
: *
dtype0?
concatenate/concatConcatV2)batch_normalization_1/batchnorm/add_1:z:0re_lu_1/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????T?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:T?
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource^dense_2/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@*
T0q
 dense_2/kernel/Regularizer/ConstConst*
dtype0*
valueB"       *
_output_shapes
:?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource^dense/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0o
dense/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource^dense_1/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource^dense_3/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@@*
T0q
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    ?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentitydense_4/Sigmoid:y:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp: : : : : : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_16603

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
t
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_15172

inputs
inputs_1
identityN
mulMulinputsinputs_1*'
_output_shapes
:?????????*
T0O
IdentityIdentitymul:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_17193=
9dense_3_kernel_regularizer_square_readvariableop_resource
identity??0dense_3/kernel/Regularizer/Square/ReadVariableOp?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_3_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 dense_3/kernel/Regularizer/ConstConst*
dtype0*
valueB"       *
_output_shapes
:?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentity"dense_3/kernel/Regularizer/add:z:01^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:  
??
?
 __inference__wrapped_model_14410
input_1?
;model_batch_normalization_batchnorm_readvariableop_resourceC
?model_batch_normalization_batchnorm_mul_readvariableop_resourceA
=model_batch_normalization_batchnorm_readvariableop_1_resourceA
=model_batch_normalization_batchnorm_readvariableop_2_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resourceA
=model_batch_normalization_2_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_2_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_2_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_2_batchnorm_readvariableop_2_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resourceA
=model_batch_normalization_3_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_3_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_3_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_3_batchnorm_readvariableop_2_resourceA
=model_batch_normalization_1_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_1_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_1_batchnorm_readvariableop_2_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource
identity??2model/batch_normalization/batchnorm/ReadVariableOp?4model/batch_normalization/batchnorm/ReadVariableOp_1?4model/batch_normalization/batchnorm/ReadVariableOp_2?6model/batch_normalization/batchnorm/mul/ReadVariableOp?4model/batch_normalization_1/batchnorm/ReadVariableOp?6model/batch_normalization_1/batchnorm/ReadVariableOp_1?6model/batch_normalization_1/batchnorm/ReadVariableOp_2?8model/batch_normalization_1/batchnorm/mul/ReadVariableOp?4model/batch_normalization_2/batchnorm/ReadVariableOp?6model/batch_normalization_2/batchnorm/ReadVariableOp_1?6model/batch_normalization_2/batchnorm/ReadVariableOp_2?8model/batch_normalization_2/batchnorm/mul/ReadVariableOp?4model/batch_normalization_3/batchnorm/ReadVariableOp?6model/batch_normalization_3/batchnorm/ReadVariableOp_1?6model/batch_normalization_3/batchnorm/ReadVariableOp_2?8model/batch_normalization_3/batchnorm/mul/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?$model/dense_3/BiasAdd/ReadVariableOp?#model/dense_3/MatMul/ReadVariableOp?$model/dense_4/BiasAdd/ReadVariableOp?#model/dense_4/MatMul/ReadVariableOph
&model/batch_normalization/LogicalAnd/xConst*
value	B
 Z *
_output_shapes
: *
dtype0
h
&model/batch_normalization/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z?
$model/batch_normalization/LogicalAnd
LogicalAnd/model/batch_normalization/LogicalAnd/x:output:0/model/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: ?
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o?:*
dtype0?
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
_output_shapes
:*
T0?
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
)model/batch_normalization/batchnorm/mul_1Mulinput_1+model/batch_normalization/batchnorm/mul:z:0*'
_output_shapes
:?????????*
T0?
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:?
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
_output_shapes
:*
T0?
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
model/dense_2/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
(model/batch_normalization_2/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z j
(model/batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
&model/batch_normalization_2/LogicalAnd
LogicalAnd1model/batch_normalization_2/LogicalAnd/x:output:01model/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: ?
4model/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_2_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@p
+model/batch_normalization_2/batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0?
)model/batch_normalization_2/batchnorm/addAddV2<model/batch_normalization_2/batchnorm/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
_output_shapes
:@*
T0?
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@?
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_2_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:0@model/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:@*
T0?
+model/batch_normalization_2/batchnorm/mul_1Mulmodel/dense_2/BiasAdd:output:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
+model/batch_normalization_2/batchnorm/mul_2Mul>model/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
_output_shapes
:@*
T0?
6model/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
)model/batch_normalization_2/batchnorm/subSub>model/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@?
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
model/dense/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0)model/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
model/re_lu/ReluRelu/model/batch_normalization_2/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
T0?
model/tf_op_layer_mul/mulMul-model/batch_normalization/batchnorm/add_1:z:0model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
model/dense_1/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
model/dense_3/MatMulMatMulmodel/re_lu/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
model/tf_op_layer_mul_1/mul_1Mul-model/batch_normalization/batchnorm/add_1:z:0model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
model/tf_op_layer_add/addAddV2model/tf_op_layer_mul/mul:z:0-model/batch_normalization/batchnorm/add_1:z:0*'
_output_shapes
:?????????*
T0j
(model/batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
j
(model/batch_normalization_3/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ?
&model/batch_normalization_3/LogicalAnd
LogicalAnd1model/batch_normalization_3/LogicalAnd/x:output:01model/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: ?
4model/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_3_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0p
+model/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
)model/batch_normalization_3/batchnorm/addAddV2<model/batch_normalization_3/batchnorm/ReadVariableOp:value:04model/batch_normalization_3/batchnorm/add/y:output:0*
_output_shapes
:@*
T0?
+model/batch_normalization_3/batchnorm/RsqrtRsqrt-model/batch_normalization_3/batchnorm/add:z:0*
_output_shapes
:@*
T0?
8model/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_3_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
)model/batch_normalization_3/batchnorm/mulMul/model/batch_normalization_3/batchnorm/Rsqrt:y:0@model/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
+model/batch_normalization_3/batchnorm/mul_1Mulmodel/dense_3/BiasAdd:output:0-model/batch_normalization_3/batchnorm/mul:z:0*'
_output_shapes
:?????????@*
T0?
6model/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_3_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
+model/batch_normalization_3/batchnorm/mul_2Mul>model/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_3/batchnorm/mul:z:0*
_output_shapes
:@*
T0?
6model/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_3_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
)model/batch_normalization_3/batchnorm/subSub>model/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@?
+model/batch_normalization_3/batchnorm/add_1AddV2/model/batch_normalization_3/batchnorm/mul_1:z:0-model/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
model/tf_op_layer_add_1/add_1AddV2!model/tf_op_layer_mul_1/mul_1:z:0model/tf_op_layer_add/add:z:0*'
_output_shapes
:?????????*
T0j
(model/batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z j
(model/batch_normalization_1/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z?
&model/batch_normalization_1/LogicalAnd
LogicalAnd1model/batch_normalization_1/LogicalAnd/x:output:01model/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: ?
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: ?
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
_output_shapes
:*
T0?
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:?
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
+model/batch_normalization_1/batchnorm/mul_1Mul!model/tf_op_layer_add_1/add_1:z:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
_output_shapes
:*
T0?
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????}
model/re_lu_1/ReluRelu/model/batch_normalization_3/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
T0_
model/concatenate/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0?
model/concatenate/concatConcatV2/model/batch_normalization_1/batchnorm/add_1:z:0 model/re_lu_1/Relu:activations:0&model/concatenate/concat/axis:output:0*
T0*'
_output_shapes
:?????????T*
N?
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:T*
dtype0?
model/dense_4/MatMulMatMul!model/concatenate/concat:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model/dense_4/SigmoidSigmoidmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:??????????

IdentityIdentitymodel/dense_4/Sigmoid:y:03^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp5^model/batch_normalization_2/batchnorm/ReadVariableOp7^model/batch_normalization_2/batchnorm/ReadVariableOp_17^model/batch_normalization_2/batchnorm/ReadVariableOp_29^model/batch_normalization_2/batchnorm/mul/ReadVariableOp5^model/batch_normalization_3/batchnorm/ReadVariableOp7^model/batch_normalization_3/batchnorm/ReadVariableOp_17^model/batch_normalization_3/batchnorm/ReadVariableOp_29^model/batch_normalization_3/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2p
6model/batch_normalization_2/batchnorm/ReadVariableOp_16model/batch_normalization_2/batchnorm/ReadVariableOp_12p
6model/batch_normalization_2/batchnorm/ReadVariableOp_26model/batch_normalization_2/batchnorm/ReadVariableOp_22L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2t
8model/batch_normalization_3/batchnorm/mul/ReadVariableOp8model/batch_normalization_3/batchnorm/mul/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2l
4model/batch_normalization_3/batchnorm/ReadVariableOp4model/batch_normalization_3/batchnorm/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2p
6model/batch_normalization_3/batchnorm/ReadVariableOp_16model/batch_normalization_3/batchnorm/ReadVariableOp_12p
6model/batch_normalization_3/batchnorm/ReadVariableOp_26model/batch_normalization_3/batchnorm/ReadVariableOp_22h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12l
4model/batch_normalization_2/batchnorm/ReadVariableOp4model/batch_normalization_2/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22t
8model/batch_normalization_2/batchnorm/mul/ReadVariableOp8model/batch_normalization_2/batchnorm/mul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp: : : : : : : : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
?
]
1__inference_tf_op_layer_mul_1_layer_call_fn_16797
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15269*U
fPRN
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_15262*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
3__inference_batch_normalization_layer_call_fn_16535

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-14524*
Tout
2*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14523*'
_output_shapes
:?????????*
Tin	
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16526

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????*
T0?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes
:*
T0?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2:& "
 
_user_specified_nameinputs: : : : 
?|
?
@__inference_model_layer_call_and_return_conditional_losses_15462
input_16
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_48
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_12batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tout
2*
Tin	
2*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14523**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-14524?
dense_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????@*
Tout
2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15073*,
_gradient_op_typePartitionedCall-15079*
Tin
2?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-14678**
config_proto

GPU 

CPU2J 8*
Tin	
2*
Tout
2*'
_output_shapes
:?????????@*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14677?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15131*
Tout
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15137*
Tin
2?
re_lu/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_15153*
Tout
2*'
_output_shapes
:?????????@*,
_gradient_op_typePartitionedCall-15159**
config_proto

GPU 

CPU2J 8?
tf_op_layer_mul/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-15179*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_15172*'
_output_shapes
:?????????*
Tout
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15210*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15204*'
_output_shapes
:?????????*
Tout
2?
dense_3/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*,
_gradient_op_typePartitionedCall-15245*
Tout
2*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15239*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8?
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*'
_output_shapes
:?????????*
Tout
2*
Tin
2*U
fPRN
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_15262*,
_gradient_op_typePartitionedCall-15269**
config_proto

GPU 

CPU2J 8?
tf_op_layer_add/PartitionedCallPartitionedCall(tf_op_layer_mul/PartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*'
_output_shapes
:?????????*
Tin
2*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_15282*
Tout
2*,
_gradient_op_typePartitionedCall-15289**
config_proto

GPU 

CPU2J 8?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14832*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14831*'
_output_shapes
:?????????@*
Tout
2*
Tin	
2?
!tf_op_layer_add_1/PartitionedCallPartitionedCall*tf_op_layer_mul_1/PartitionedCall:output:0(tf_op_layer_add/PartitionedCall:output:0*'
_output_shapes
:?????????*U
fPRN
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_15325*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-15332?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*tf_op_layer_add_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*
Tin	
2*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14985*,
_gradient_op_typePartitionedCall-14986*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tout
2?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-15373*K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_15367*
Tin
2*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*
Tout
2?
concatenate/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_15387*,
_gradient_op_typePartitionedCall-15394*'
_output_shapes
:?????????T?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*,
_gradient_op_typePartitionedCall-15418*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15412*'
_output_shapes
:?????????*
Tout
2**
config_proto

GPU 

CPU2J 8?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_statefulpartitionedcall_args_1 ^dense_2/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
 dense_2/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_statefulpartitionedcall_args_1^dense/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:o
dense/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_statefulpartitionedcall_args_1 ^dense_1/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0q
 dense_1/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_statefulpartitionedcall_args_1 ^dense_3/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@@*
T0q
 dense_3/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_16570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
C
'__inference_re_lu_1_layer_call_fn_17104

inputs
identity?
PartitionedCallPartitionedCallinputs*K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_15367*,
_gradient_op_typePartitionedCall-15373*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????@`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?7
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14831

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
_output_shapes

:@*
T0*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
_output_shapes

:@*
T0*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
squeeze_dims
 *
_output_shapes
:@s
moments/Squeeze_1Squeezemoments/variance:output:0*
squeeze_dims
 *
T0*
_output_shapes
:@?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 ?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: ?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpT
batchnorm/add/yConst*
valueB
 *o?:*
dtype0*
_output_shapes
: q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
_output_shapes
:@*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????@*
T0h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????@*
T0?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_16756

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16956

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@T
batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes
:@*
T0?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:@*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????@*
T0?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_15204

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
%__inference_model_layer_call_fn_16424

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tout
2*,
_gradient_op_typePartitionedCall-15736*&
Tin
2*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15735?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
?
'__inference_dense_1_layer_call_fn_16763

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-15210*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15204*
Tout
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
5__inference_batch_normalization_2_layer_call_fn_16721

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-14678*
Tin	
2*
Tout
2*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14677?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
v
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_16769
inputs_0
inputs_1
identityP
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
%__inference_model_layer_call_fn_16393

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15624*&
Tin
2*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15623?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
??
?
@__inference_model_layer_call_and_return_conditional_losses_16211

inputsD
@batch_normalization_assignmovingavg_read_readvariableop_resourceF
Bbatch_normalization_assignmovingavg_1_read_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resourceF
Bbatch_normalization_2_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resourceF
Bbatch_normalization_3_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resourceF
Bbatch_normalization_1_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??7batch_normalization/AssignMovingAvg/AssignSubVariableOp?7batch_normalization/AssignMovingAvg/Read/ReadVariableOp?2batch_normalization/AssignMovingAvg/ReadVariableOp?9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp?9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp?;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?2batch_normalization_1/batchnorm/mul/ReadVariableOp?9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp?9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp?;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_2/batchnorm/ReadVariableOp?2batch_normalization_2/batchnorm/mul/ReadVariableOp?9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp?9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp?4batch_normalization_3/AssignMovingAvg/ReadVariableOp?;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp?;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp?6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_3/batchnorm/ReadVariableOp?2batch_normalization_3/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOpb
 batch_normalization/LogicalAnd/xConst*
dtype0
*
value	B
 Z*
_output_shapes
: b
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
?
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: |
2batch_normalization/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:?
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:?
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*'
_output_shapes
:?????????*
T0?
6batch_normalization/moments/variance/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
squeeze_dims
 *
T0*
_output_shapes
:?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
_output_shapes
:*
T0*
squeeze_dims
 ?
7batch_normalization/AssignMovingAvg/Read/ReadVariableOpReadVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
,batch_normalization/AssignMovingAvg/IdentityIdentity?batch_normalization/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0?
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource8^batch_normalization/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp?
9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
.batch_normalization/AssignMovingAvg_1/IdentityIdentityAbatch_normalization/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:?
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
valueB
 *
?#<?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource:^batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
T0*
_output_shapes
:?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOph
#batch_normalization/batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:*
T0?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*'
_output_shapes
:?????????*
T0?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
"batch_normalization_2/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: d
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: ~
4batch_normalization_2/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: ?
"batch_normalization_2/moments/meanMeandense_2/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
	keep_dims(*
_output_shapes

:@?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
_output_shapes

:@*
T0?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_2/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*'
_output_shapes
:?????????@*
T0?
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
_output_shapes
:@*
T0*
squeeze_dims
 ?
9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_2_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
.batch_normalization_2/AssignMovingAvg/IdentityIdentityAbatch_normalization_2/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0?
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*
_output_shapes
: *
dtype0*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_2_assignmovingavg_read_readvariableop_resource:^batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp?
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_2_assignmovingavg_read_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ?
;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
0batch_normalization_2/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
valueB
 *
?#<?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp*
T0*
_output_shapes
:@?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
T0*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp?
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOpj
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
%batch_normalization_2/batchnorm/mul_1Muldense_2/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
_output_shapes
:@*
T0?
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
_output_shapes
:@*
T0?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????o

re_lu/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
T0?
tf_op_layer_mul/mulMul'batch_normalization/batchnorm/add_1:z:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
dense_1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
dense_3/MatMulMatMulre_lu/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
tf_op_layer_mul_1/mul_1Mul'batch_normalization/batchnorm/add_1:z:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
tf_op_layer_add/addAddV2tf_op_layer_mul/mul:z:0'batch_normalization/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????d
"batch_normalization_3/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: d
"batch_normalization_3/LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ?
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: ~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_3/moments/meanMeandense_3/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(?
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@?
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_3/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*'
_output_shapes
:?????????@*
T0?
8batch_normalization_3/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:?
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
_output_shapes

:@*
T0*
	keep_dims(?
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0?
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 ?
9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_3_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
.batch_normalization_3/AssignMovingAvg/IdentityIdentityAbatch_normalization_3/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*
_output_shapes
: *
dtype0*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_3_assignmovingavg_read_readvariableop_resource:^batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
T0?
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_3_assignmovingavg_read_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 ?
;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
0batch_normalization_3/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: *N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
valueB
 *
?#<?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0?
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 j
%batch_normalization_3/batchnorm/add/yConst*
valueB
 *o?:*
dtype0*
_output_shapes
: ?
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
_output_shapes
:@*
T0|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
_output_shapes
:@*
T0?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
%batch_normalization_3/batchnorm/mul_1Muldense_3/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
_output_shapes
:@*
T0?
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
_output_shapes
:@*
T0?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
tf_op_layer_add_1/add_1AddV2tf_op_layer_mul_1/mul_1:z:0tf_op_layer_add/add:z:0*'
_output_shapes
:?????????*
T0d
"batch_normalization_1/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Zd
"batch_normalization_1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ?
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: ~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_1/moments/meanMeantf_op_layer_add_1/add_1:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
	keep_dims(*
_output_shapes

:*
T0?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
_output_shapes

:*
T0?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencetf_op_layer_add_1/add_1:z:03batch_normalization_1/moments/StopGradient:output:0*'
_output_shapes
:?????????*
T0?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
squeeze_dims
 *
_output_shapes
:*
T0?
9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_1_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
.batch_normalization_1/AssignMovingAvg/IdentityIdentityAbatch_normalization_1/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0?
+batch_normalization_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: ?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_1_assignmovingavg_read_readvariableop_resource:^batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_1_assignmovingavg_read_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 ?
;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
0batch_normalization_1/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:?
-batch_normalization_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
T0?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
_output_shapes
:*
T0|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
_output_shapes
:*
T0?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:*
T0?
%batch_normalization_1/batchnorm/mul_1Multf_op_layer_add_1/add_1:z:0'batch_normalization_1/batchnorm/mul:z:0*'
_output_shapes
:?????????*
T0?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*'
_output_shapes
:?????????*
T0q
re_lu_1/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????@Y
concatenate/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :?
concatenate/concatConcatV2)batch_normalization_1/batchnorm/add_1:z:0re_lu_1/Relu:activations:0 concatenate/concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????T?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:T*
dtype0?
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource^dense_2/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@*
T0q
 dense_2/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource^dense/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource^dense_1/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource^dense_3/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@@*
T0q
 dense_3/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentitydense_4/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp8^batch_normalization/AssignMovingAvg/Read/ReadVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/Read/ReadVariableOp7batch_normalization/AssignMovingAvg/Read/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2v
9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp: : : : : : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : 
?
?
5__inference_batch_normalization_1_layer_call_fn_17094

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_15020*
Tout
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15021*
Tin	
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
?	
?
B__inference_dense_4_layer_call_and_return_conditional_losses_17128

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:T*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
[
/__inference_tf_op_layer_mul_layer_call_fn_16775
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_15172*,
_gradient_op_typePartitionedCall-15179`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
'__inference_dense_4_layer_call_fn_17135

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15412*,
_gradient_op_typePartitionedCall-15418*
Tout
2*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????T::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_15020

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:T
batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
_output_shapes
:*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes
:*
T0?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:*
T0c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes
:*
T0?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2:& "
 
_user_specified_nameinputs: : : : 
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14558

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes
:*
T0?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????*
T0?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes
:*
T0?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2: : : :& "
 
_user_specified_nameinputs: 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_15073

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
5__inference_batch_normalization_3_layer_call_fn_16965

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-14832*
Tin	
2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????@*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14831?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
%__inference_model_layer_call_fn_15653
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15623*&
Tin
2*,
_gradient_op_typePartitionedCall-15624*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
W
+__inference_concatenate_layer_call_fn_17117
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_15387**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????T*,
_gradient_op_typePartitionedCall-15394*
Tout
2*
Tin
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????T*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?7
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14677

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
value	B
 Z*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*'
_output_shapes
:?????????@*
T0l
"moments/variance/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*
_output_shapes

:@*
T0m
moments/SqueezeSqueezemoments/mean:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0s
moments/Squeeze_1Squeezemoments/variance:output:0*
_output_shapes
:@*
squeeze_dims
 *
T0?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *
?#<*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 *
dtype0?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *
?#<*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0T
batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
_output_shapes
:@*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes
:@*
T0?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
_output_shapes
:@*
T0?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
_output_shapes
:@*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????@*
T0?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_15239

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????@*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@@*
T0q
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????@::2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
??
?*
!__inference__traced_restore_17647
file_prefix.
*assignvariableop_batch_normalization_gamma/
+assignvariableop_1_batch_normalization_beta6
2assignvariableop_2_batch_normalization_moving_mean:
6assignvariableop_3_batch_normalization_moving_variance%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias2
.assignvariableop_8_batch_normalization_2_gamma1
-assignvariableop_9_batch_normalization_2_beta9
5assignvariableop_10_batch_normalization_2_moving_mean=
9assignvariableop_11_batch_normalization_2_moving_variance&
"assignvariableop_12_dense_1_kernel$
 assignvariableop_13_dense_1_bias&
"assignvariableop_14_dense_3_kernel$
 assignvariableop_15_dense_3_bias3
/assignvariableop_16_batch_normalization_3_gamma2
.assignvariableop_17_batch_normalization_3_beta9
5assignvariableop_18_batch_normalization_3_moving_mean=
9assignvariableop_19_batch_normalization_3_moving_variance3
/assignvariableop_20_batch_normalization_1_gamma2
.assignvariableop_21_batch_normalization_1_beta9
5assignvariableop_22_batch_normalization_1_moving_mean=
9assignvariableop_23_batch_normalization_1_moving_variance&
"assignvariableop_24_dense_4_kernel$
 assignvariableop_25_dense_4_bias,
(assignvariableop_26_training_adamax_iter.
*assignvariableop_27_training_adamax_beta_1.
*assignvariableop_28_training_adamax_beta_2-
)assignvariableop_29_training_adamax_decay5
1assignvariableop_30_training_adamax_learning_rate
assignvariableop_31_total
assignvariableop_32_countC
?assignvariableop_33_training_adamax_batch_normalization_gamma_mB
>assignvariableop_34_training_adamax_batch_normalization_beta_m8
4assignvariableop_35_training_adamax_dense_2_kernel_m6
2assignvariableop_36_training_adamax_dense_2_bias_m6
2assignvariableop_37_training_adamax_dense_kernel_m4
0assignvariableop_38_training_adamax_dense_bias_mE
Aassignvariableop_39_training_adamax_batch_normalization_2_gamma_mD
@assignvariableop_40_training_adamax_batch_normalization_2_beta_m8
4assignvariableop_41_training_adamax_dense_1_kernel_m6
2assignvariableop_42_training_adamax_dense_1_bias_m8
4assignvariableop_43_training_adamax_dense_3_kernel_m6
2assignvariableop_44_training_adamax_dense_3_bias_mE
Aassignvariableop_45_training_adamax_batch_normalization_3_gamma_mD
@assignvariableop_46_training_adamax_batch_normalization_3_beta_mE
Aassignvariableop_47_training_adamax_batch_normalization_1_gamma_mD
@assignvariableop_48_training_adamax_batch_normalization_1_beta_m8
4assignvariableop_49_training_adamax_dense_4_kernel_m6
2assignvariableop_50_training_adamax_dense_4_bias_mC
?assignvariableop_51_training_adamax_batch_normalization_gamma_vB
>assignvariableop_52_training_adamax_batch_normalization_beta_v8
4assignvariableop_53_training_adamax_dense_2_kernel_v6
2assignvariableop_54_training_adamax_dense_2_bias_v6
2assignvariableop_55_training_adamax_dense_kernel_v4
0assignvariableop_56_training_adamax_dense_bias_vE
Aassignvariableop_57_training_adamax_batch_normalization_2_gamma_vD
@assignvariableop_58_training_adamax_batch_normalization_2_beta_v8
4assignvariableop_59_training_adamax_dense_1_kernel_v6
2assignvariableop_60_training_adamax_dense_1_bias_v8
4assignvariableop_61_training_adamax_dense_3_kernel_v6
2assignvariableop_62_training_adamax_dense_3_bias_vE
Aassignvariableop_63_training_adamax_batch_normalization_3_gamma_vD
@assignvariableop_64_training_adamax_batch_normalization_3_beta_vE
Aassignvariableop_65_training_adamax_batch_normalization_1_gamma_vD
@assignvariableop_66_training_adamax_batch_normalization_1_beta_v8
4assignvariableop_67_training_adamax_dense_4_kernel_v6
2assignvariableop_68_training_adamax_dense_4_bias_v
identity_70??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*?%
value?%B?%EB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*?
value?B?EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*S
dtypesI
G2E	*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0?
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0}
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_2_gammaIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_2_betaIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_2_moving_meanIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_2_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_3_kernelIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_3_biasIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_3_gammaIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_3_betaIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_3_moving_meanIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_3_moving_varianceIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_1_gammaIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_1_betaIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_1_moving_meanIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_1_moving_varianceIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_4_kernelIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0?
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_4_biasIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0	?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_training_adamax_iterIdentity_26:output:0*
_output_shapes
 *
dtype0	P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_training_adamax_beta_1Identity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_training_adamax_beta_2Identity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_training_adamax_decayIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0?
AssignVariableOp_30AssignVariableOp1assignvariableop_30_training_adamax_learning_rateIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:{
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:{
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0?
AssignVariableOp_33AssignVariableOp?assignvariableop_33_training_adamax_batch_normalization_gamma_mIdentity_33:output:0*
_output_shapes
 *
dtype0P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_training_adamax_batch_normalization_beta_mIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_training_adamax_dense_2_kernel_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp2assignvariableop_36_training_adamax_dense_2_bias_mIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T0?
AssignVariableOp_37AssignVariableOp2assignvariableop_37_training_adamax_dense_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype0P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_training_adamax_dense_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype0P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpAassignvariableop_39_training_adamax_batch_normalization_2_gamma_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0?
AssignVariableOp_40AssignVariableOp@assignvariableop_40_training_adamax_batch_normalization_2_beta_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
_output_shapes
:*
T0?
AssignVariableOp_41AssignVariableOp4assignvariableop_41_training_adamax_dense_1_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp2assignvariableop_42_training_adamax_dense_1_bias_mIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp4assignvariableop_43_training_adamax_dense_3_kernel_mIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
_output_shapes
:*
T0?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_training_adamax_dense_3_bias_mIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0?
AssignVariableOp_45AssignVariableOpAassignvariableop_45_training_adamax_batch_normalization_3_gamma_mIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
_output_shapes
:*
T0?
AssignVariableOp_46AssignVariableOp@assignvariableop_46_training_adamax_batch_normalization_3_beta_mIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
_output_shapes
:*
T0?
AssignVariableOp_47AssignVariableOpAassignvariableop_47_training_adamax_batch_normalization_1_gamma_mIdentity_47:output:0*
_output_shapes
 *
dtype0P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp@assignvariableop_48_training_adamax_batch_normalization_1_beta_mIdentity_48:output:0*
_output_shapes
 *
dtype0P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp4assignvariableop_49_training_adamax_dense_4_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype0P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0?
AssignVariableOp_50AssignVariableOp2assignvariableop_50_training_adamax_dense_4_bias_mIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0?
AssignVariableOp_51AssignVariableOp?assignvariableop_51_training_adamax_batch_normalization_gamma_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0?
AssignVariableOp_52AssignVariableOp>assignvariableop_52_training_adamax_batch_normalization_beta_vIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp4assignvariableop_53_training_adamax_dense_2_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype0P
Identity_54IdentityRestoreV2:tensors:54*
_output_shapes
:*
T0?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_training_adamax_dense_2_bias_vIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_training_adamax_dense_kernel_vIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
_output_shapes
:*
T0?
AssignVariableOp_56AssignVariableOp0assignvariableop_56_training_adamax_dense_bias_vIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
_output_shapes
:*
T0?
AssignVariableOp_57AssignVariableOpAassignvariableop_57_training_adamax_batch_normalization_2_gamma_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
_output_shapes
:*
T0?
AssignVariableOp_58AssignVariableOp@assignvariableop_58_training_adamax_batch_normalization_2_beta_vIdentity_58:output:0*
_output_shapes
 *
dtype0P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp4assignvariableop_59_training_adamax_dense_1_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype0P
Identity_60IdentityRestoreV2:tensors:60*
_output_shapes
:*
T0?
AssignVariableOp_60AssignVariableOp2assignvariableop_60_training_adamax_dense_1_bias_vIdentity_60:output:0*
dtype0*
_output_shapes
 P
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp4assignvariableop_61_training_adamax_dense_3_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype0P
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp2assignvariableop_62_training_adamax_dense_3_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype0P
Identity_63IdentityRestoreV2:tensors:63*
_output_shapes
:*
T0?
AssignVariableOp_63AssignVariableOpAassignvariableop_63_training_adamax_batch_normalization_3_gamma_vIdentity_63:output:0*
_output_shapes
 *
dtype0P
Identity_64IdentityRestoreV2:tensors:64*
_output_shapes
:*
T0?
AssignVariableOp_64AssignVariableOp@assignvariableop_64_training_adamax_batch_normalization_3_beta_vIdentity_64:output:0*
_output_shapes
 *
dtype0P
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpAassignvariableop_65_training_adamax_batch_normalization_1_gamma_vIdentity_65:output:0*
dtype0*
_output_shapes
 P
Identity_66IdentityRestoreV2:tensors:66*
_output_shapes
:*
T0?
AssignVariableOp_66AssignVariableOp@assignvariableop_66_training_adamax_batch_normalization_1_beta_vIdentity_66:output:0*
dtype0*
_output_shapes
 P
Identity_67IdentityRestoreV2:tensors:67*
_output_shapes
:*
T0?
AssignVariableOp_67AssignVariableOp4assignvariableop_67_training_adamax_dense_4_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype0P
Identity_68IdentityRestoreV2:tensors:68*
_output_shapes
:*
T0?
AssignVariableOp_68AssignVariableOp2assignvariableop_68_training_adamax_dense_4_bias_vIdentity_68:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?
Identity_70IdentityIdentity_69:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_70Identity_70:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E 
?
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_15367

inputs
identityF
ReluReluinputs*'
_output_shapes
:?????????@*
T0Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
x
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_16848
inputs_0
inputs_1
identityT
add_1AddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	add_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_15387

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*'
_output_shapes
:?????????T*
T0W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????T"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
v
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_15325

inputs
inputs_1
identityR
add_1AddV2inputsinputs_1*
T0*'
_output_shapes
:?????????Q
IdentityIdentity	add_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14866

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z N
LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
_output_shapes
:@*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:@*
T0c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????@*
T0?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes
:@*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2:& "
 
_user_specified_nameinputs: : : : 
?
r
F__inference_concatenate_layer_call_and_return_conditional_losses_17111
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
value	B :*
dtype0w
concatConcatV2inputs_0inputs_1concat/axis:output:0*'
_output_shapes
:?????????T*
T0*
NW
IdentityIdentityconcat:output:0*'
_output_shapes
:?????????T*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?|
?
@__inference_model_layer_call_and_return_conditional_losses_15542
input_16
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_48
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_12batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14558*,
_gradient_op_typePartitionedCall-14559*
Tin	
2*'
_output_shapes
:?????????*
Tout
2**
config_proto

GPU 

CPU2J 8?
dense_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tout
2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15073**
config_proto

GPU 

CPU2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-15079*'
_output_shapes
:?????????@?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*'
_output_shapes
:?????????@*
Tout
2*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14712**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14713?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15131*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15137?
re_lu/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-15159*I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_15153*'
_output_shapes
:?????????@*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2?
tf_op_layer_mul/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense/StatefulPartitionedCall:output:0*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15179*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_15172*'
_output_shapes
:?????????*
Tout
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15204*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15210?
dense_3/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????@*
Tout
2*,
_gradient_op_typePartitionedCall-15245*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15239**
config_proto

GPU 

CPU2J 8?
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*U
fPRN
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_15262*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15269**
config_proto

GPU 

CPU2J 8*
Tout
2?
tf_op_layer_add/PartitionedCallPartitionedCall(tf_op_layer_mul/PartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15289*
Tout
2*
Tin
2*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_15282**
config_proto

GPU 

CPU2J 8?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*'
_output_shapes
:?????????@*
Tout
2*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14866*
Tin	
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-14867?
!tf_op_layer_add_1/PartitionedCallPartitionedCall*tf_op_layer_mul_1/PartitionedCall:output:0(tf_op_layer_add/PartitionedCall:output:0*
Tout
2*
Tin
2*U
fPRN
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_15325*,
_gradient_op_typePartitionedCall-15332**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:??????????
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*tf_op_layer_add_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_15020*
Tin	
2*,
_gradient_op_typePartitionedCall-15021*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tout
2?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15373*
Tin
2*K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_15367*'
_output_shapes
:?????????@?
concatenate/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????T*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_15387*,
_gradient_op_typePartitionedCall-15394*
Tout
2?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tout
2*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15412*
Tin
2**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15418*'
_output_shapes
:??????????
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_statefulpartitionedcall_args_1 ^dense_2/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@*
T0q
 dense_2/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: ?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_statefulpartitionedcall_args_1^dense/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_statefulpartitionedcall_args_1 ^dense_1/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0q
 dense_1/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_statefulpartitionedcall_args_1 ^dense_3/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 dense_3/kernel/Regularizer/ConstConst*
dtype0*
valueB"       *
_output_shapes
:?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
?
__inference_loss_fn_0_17148=
9dense_2_kernel_regularizer_square_readvariableop_resource
identity??0dense_2/kernel/Regularizer/Square/ReadVariableOp?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
?#<*
dtype0?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentity"dense_2/kernel/Regularizer/add:z:01^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:  
?
?
5__inference_batch_normalization_3_layer_call_fn_16974

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-14867*
Tin	
2**
config_proto

GPU 

CPU2J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14866*'
_output_shapes
:?????????@*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?|
?
@__inference_model_layer_call_and_return_conditional_losses_15735

inputs6
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_48
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?0dense_3/kernel/Regularizer/Square/ReadVariableOp?dense_4/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs2batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tout
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-14559*
Tin	
2**
config_proto

GPU 

CPU2J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14558?
dense_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????@*,
_gradient_op_typePartitionedCall-15079*
Tin
2*
Tout
2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15073?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tout
2*,
_gradient_op_typePartitionedCall-14713*
Tin	
2**
config_proto

GPU 

CPU2J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14712*'
_output_shapes
:?????????@?
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15131*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15137*
Tout
2**
config_proto

GPU 

CPU2J 8?
re_lu/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_15153*,
_gradient_op_typePartitionedCall-15159*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????@*
Tout
2?
tf_op_layer_mul/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense/StatefulPartitionedCall:output:0*
Tout
2*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-15179*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_15172**
config_proto

GPU 

CPU2J 8?
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-15210*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15204*'
_output_shapes
:?????????*
Tin
2?
dense_3/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????@*
Tout
2*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15239**
config_proto

GPU 

CPU2J 8*,
_gradient_op_typePartitionedCall-15245?
!tf_op_layer_mul_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*U
fPRN
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_15262*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15269?
tf_op_layer_add/PartitionedCallPartitionedCall(tf_op_layer_mul/PartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-15289*'
_output_shapes
:?????????*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_15282?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_14866*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*,
_gradient_op_typePartitionedCall-14867*'
_output_shapes
:?????????@?
!tf_op_layer_add_1/PartitionedCallPartitionedCall*tf_op_layer_mul_1/PartitionedCall:output:0(tf_op_layer_add/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-15332*U
fPRN
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_15325*
Tin
2?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*tf_op_layer_add_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-15021*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_15020**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin	
2*
Tout
2?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:?????????@*,
_gradient_op_typePartitionedCall-15373*K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_15367?
concatenate/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tout
2*
Tin
2*'
_output_shapes
:?????????T*,
_gradient_op_typePartitionedCall-15394**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_15387?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-15418*
Tout
2*'
_output_shapes
:?????????*
Tin
2*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15412**
config_proto

GPU 

CPU2J 8?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_statefulpartitionedcall_args_1 ^dense_2/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@*
T0q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_2/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/add/x:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_statefulpartitionedcall_args_1^dense/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0o
dense/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       ?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *
?#<*
_output_shapes
: ?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_statefulpartitionedcall_args_1 ^dense_1/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:*
T0q
 dense_1/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_statefulpartitionedcall_args_1 ^dense_3/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@@*
dtype0?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes

:@@*
T0q
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/mul/xConst*
valueB
 *
?#<*
_output_shapes
: *
dtype0?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17099

inputs
identityF
ReluReluinputs*'
_output_shapes
:?????????@*
T0Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_15986
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*,
_gradient_op_typePartitionedCall-15957*)
f$R"
 __inference__wrapped_model_14410*&
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
v
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_15262

inputs
inputs_1
identityP
mul_1Mulinputsinputs_1*'
_output_shapes
:?????????*
T0Q
IdentityIdentity	mul_1:z:0*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_16544

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14558*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-14559*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
A
%__inference_re_lu_layer_call_fn_16785

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*,
_gradient_op_typePartitionedCall-15159**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_15153*'
_output_shapes
:?????????@*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????@*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?7
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14523

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
_output_shapes

:*
T0*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
squeeze_dims
 *
_output_shapes
:*
T0s
moments/Squeeze_1Squeezemoments/variance:output:0*
squeeze_dims
 *
_output_shapes
:*
T0?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
valueB
 *
?#<*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *
?#<*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 T
batchnorm/add/yConst*
valueB
 *o?:*
_output_shapes
: *
dtype0q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
_output_shapes
:*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:*
T0c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????*
T0h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
_output_shapes
:*
T0?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
_output_shapes
:*
T0r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?
v
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_16803
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17076

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
value	B
 Z *
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: ?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????*
T0?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2: :& "
 
_user_specified_nameinputs: : : 
?
?
__inference_loss_fn_1_17163;
7dense_kernel_regularizer_square_readvariableop_resource
identity??.dense/kernel/Regularizer/Square/ReadVariableOp?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
?#<?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0?
IdentityIdentity dense/kernel/Regularizer/add:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:  
ǃ
?"
__inference__traced_save_17427
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop3
/savev2_training_adamax_iter_read_readvariableop	5
1savev2_training_adamax_beta_1_read_readvariableop5
1savev2_training_adamax_beta_2_read_readvariableop4
0savev2_training_adamax_decay_read_readvariableop<
8savev2_training_adamax_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopJ
Fsavev2_training_adamax_batch_normalization_gamma_m_read_readvariableopI
Esavev2_training_adamax_batch_normalization_beta_m_read_readvariableop?
;savev2_training_adamax_dense_2_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_2_bias_m_read_readvariableop=
9savev2_training_adamax_dense_kernel_m_read_readvariableop;
7savev2_training_adamax_dense_bias_m_read_readvariableopL
Hsavev2_training_adamax_batch_normalization_2_gamma_m_read_readvariableopK
Gsavev2_training_adamax_batch_normalization_2_beta_m_read_readvariableop?
;savev2_training_adamax_dense_1_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_1_bias_m_read_readvariableop?
;savev2_training_adamax_dense_3_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_3_bias_m_read_readvariableopL
Hsavev2_training_adamax_batch_normalization_3_gamma_m_read_readvariableopK
Gsavev2_training_adamax_batch_normalization_3_beta_m_read_readvariableopL
Hsavev2_training_adamax_batch_normalization_1_gamma_m_read_readvariableopK
Gsavev2_training_adamax_batch_normalization_1_beta_m_read_readvariableop?
;savev2_training_adamax_dense_4_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_4_bias_m_read_readvariableopJ
Fsavev2_training_adamax_batch_normalization_gamma_v_read_readvariableopI
Esavev2_training_adamax_batch_normalization_beta_v_read_readvariableop?
;savev2_training_adamax_dense_2_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_2_bias_v_read_readvariableop=
9savev2_training_adamax_dense_kernel_v_read_readvariableop;
7savev2_training_adamax_dense_bias_v_read_readvariableopL
Hsavev2_training_adamax_batch_normalization_2_gamma_v_read_readvariableopK
Gsavev2_training_adamax_batch_normalization_2_beta_v_read_readvariableop?
;savev2_training_adamax_dense_1_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_1_bias_v_read_readvariableop?
;savev2_training_adamax_dense_3_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_3_bias_v_read_readvariableopL
Hsavev2_training_adamax_batch_normalization_3_gamma_v_read_readvariableopK
Gsavev2_training_adamax_batch_normalization_3_beta_v_read_readvariableopL
Hsavev2_training_adamax_batch_normalization_1_gamma_v_read_readvariableopK
Gsavev2_training_adamax_batch_normalization_1_beta_v_read_readvariableop?
;savev2_training_adamax_dense_4_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_4_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_21ac62cfd2ba493485c6ec372b2424d6/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
value	B :*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?&
SaveV2/tensor_namesConst"/device:CPU:0*?%
value?%B?%EB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:E*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*?
value?B?EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop/savev2_training_adamax_iter_read_readvariableop1savev2_training_adamax_beta_1_read_readvariableop1savev2_training_adamax_beta_2_read_readvariableop0savev2_training_adamax_decay_read_readvariableop8savev2_training_adamax_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopFsavev2_training_adamax_batch_normalization_gamma_m_read_readvariableopEsavev2_training_adamax_batch_normalization_beta_m_read_readvariableop;savev2_training_adamax_dense_2_kernel_m_read_readvariableop9savev2_training_adamax_dense_2_bias_m_read_readvariableop9savev2_training_adamax_dense_kernel_m_read_readvariableop7savev2_training_adamax_dense_bias_m_read_readvariableopHsavev2_training_adamax_batch_normalization_2_gamma_m_read_readvariableopGsavev2_training_adamax_batch_normalization_2_beta_m_read_readvariableop;savev2_training_adamax_dense_1_kernel_m_read_readvariableop9savev2_training_adamax_dense_1_bias_m_read_readvariableop;savev2_training_adamax_dense_3_kernel_m_read_readvariableop9savev2_training_adamax_dense_3_bias_m_read_readvariableopHsavev2_training_adamax_batch_normalization_3_gamma_m_read_readvariableopGsavev2_training_adamax_batch_normalization_3_beta_m_read_readvariableopHsavev2_training_adamax_batch_normalization_1_gamma_m_read_readvariableopGsavev2_training_adamax_batch_normalization_1_beta_m_read_readvariableop;savev2_training_adamax_dense_4_kernel_m_read_readvariableop9savev2_training_adamax_dense_4_bias_m_read_readvariableopFsavev2_training_adamax_batch_normalization_gamma_v_read_readvariableopEsavev2_training_adamax_batch_normalization_beta_v_read_readvariableop;savev2_training_adamax_dense_2_kernel_v_read_readvariableop9savev2_training_adamax_dense_2_bias_v_read_readvariableop9savev2_training_adamax_dense_kernel_v_read_readvariableop7savev2_training_adamax_dense_bias_v_read_readvariableopHsavev2_training_adamax_batch_normalization_2_gamma_v_read_readvariableopGsavev2_training_adamax_batch_normalization_2_beta_v_read_readvariableop;savev2_training_adamax_dense_1_kernel_v_read_readvariableop9savev2_training_adamax_dense_1_bias_v_read_readvariableop;savev2_training_adamax_dense_3_kernel_v_read_readvariableop9savev2_training_adamax_dense_3_bias_v_read_readvariableopHsavev2_training_adamax_batch_normalization_3_gamma_v_read_readvariableopGsavev2_training_adamax_batch_normalization_3_beta_v_read_readvariableopHsavev2_training_adamax_batch_normalization_1_gamma_v_read_readvariableopGsavev2_training_adamax_batch_normalization_1_beta_v_read_readvariableop;savev2_training_adamax_dense_4_kernel_v_read_readvariableop9savev2_training_adamax_dense_4_bias_v_read_readvariableop"/device:CPU:0*S
dtypesI
G2E	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
_output_shapes
:*
N?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::@:@:::@:@:@:@:::@@:@:@:@:@:@:::::T:: : : : : : : :::@:@:::@:@:::@@:@:@:@:::T::::@:@:::@:@:::@@:@:@:@:::T:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints: :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :+ '
%
_user_specified_namefile_prefix: : : : : : : 
?7
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16503

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?#AssignMovingAvg/Read/ReadVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?%AssignMovingAvg_1/Read/ReadVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
_output_shapes
:*
squeeze_dims
 *
T0s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
squeeze_dims
 *
_output_shapes
:?
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:*
T0?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:*
T0?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 *
dtype0?
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *
?#<*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:*
T0?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpT
batchnorm/add/yConst*
dtype0*
valueB
 *o?:*
_output_shapes
: q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
_output_shapes
:*
T0P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes
:*
T0c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*'
_output_shapes
:?????????*
T0h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*'
_output_shapes
:?????????*
T0?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: : : :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????;
dense_40
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?}
_tf_keras_model?}{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 20], "dtype": "float32", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "mul", "trainable": true, "dtype": "float32", "node_def": {"name": "mul", "op": "Mul", "input": ["batch_normalization/Identity", "dense/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_mul", "inbound_nodes": [[["batch_normalization", 0, 0, {}], ["dense", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "mul_1", "op": "Mul", "input": ["batch_normalization/Identity", "dense_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_mul_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}], ["dense_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "add", "trainable": true, "dtype": "float32", "node_def": {"name": "add", "op": "AddV2", "input": ["mul", "batch_normalization/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_add", "inbound_nodes": [[["tf_op_layer_mul", 0, 0, {}], ["batch_normalization", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "add_1", "trainable": true, "dtype": "float32", "node_def": {"name": "add_1", "op": "AddV2", "input": ["mul_1", "add"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_add_1", "inbound_nodes": [[["tf_op_layer_mul_1", 0, 0, {}], ["tf_op_layer_add", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["tf_op_layer_add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}], ["re_lu_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_4", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 20], "dtype": "float32", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "mul", "trainable": true, "dtype": "float32", "node_def": {"name": "mul", "op": "Mul", "input": ["batch_normalization/Identity", "dense/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_mul", "inbound_nodes": [[["batch_normalization", 0, 0, {}], ["dense", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "mul_1", "op": "Mul", "input": ["batch_normalization/Identity", "dense_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_mul_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}], ["dense_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "add", "trainable": true, "dtype": "float32", "node_def": {"name": "add", "op": "AddV2", "input": ["mul", "batch_normalization/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_add", "inbound_nodes": [[["tf_op_layer_mul", 0, 0, {}], ["batch_normalization", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "add_1", "trainable": true, "dtype": "float32", "node_def": {"name": "add_1", "op": "AddV2", "input": ["mul_1", "add"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_add_1", "inbound_nodes": [[["tf_op_layer_mul_1", 0, 0, {}], ["tf_op_layer_add", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["tf_op_layer_add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}], ["re_lu_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_4", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 20], "config": {"batch_input_shape": [null, 20], "dtype": "float32", "sparse": false, "name": "input_1"}}
?
axis
	gamma
beta
moving_mean
 moving_variance
!trainable_variables
"	variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 20}}}}
?

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
?

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
?
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}}
?

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
?
@	constants
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_mul", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mul", "trainable": true, "dtype": "float32", "node_def": {"name": "mul", "op": "Mul", "input": ["batch_normalization/Identity", "dense/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
I	constants
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_mul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "mul_1", "op": "Mul", "input": ["batch_normalization/Identity", "dense_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
N	constants
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32", "node_def": {"name": "add", "op": "AddV2", "input": ["mul", "batch_normalization/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
Y	constants
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32", "node_def": {"name": "add_1", "op": "AddV2", "input": ["mul_1", "add"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}}
?
gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 20}}}}
?
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}}
?

xkernel
ybias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}}
?
~iter

beta_1
?beta_2

?decay
?learning_ratem?m?%m?&m?+m?,m?2m?3m?:m?;m?Sm?Tm?_m?`m?hm?im?xm?ym?v?v?%v?&v?+v?,v?2v?3v?:v?;v?Sv?Tv?_v?`v?hv?iv?xv?yv?"
	optimizer
?
0
1
%2
&3
+4
,5
26
37
:8
;9
S10
T11
_12
`13
h14
i15
x16
y17"
trackable_list_wrapper
?
0
1
2
 3
%4
&5
+6
,7
28
39
410
511
:12
;13
S14
T15
_16
`17
a18
b19
h20
i21
j22
k23
x24
y25"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
 ?layer_regularization_losses
trainable_variables
	variables
?metrics
?non_trainable_variables
?layers
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
trainable_variables
	variables
regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
0
1"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
!trainable_variables
"	variables
#regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_2/kernel
:@2dense_2/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
'trainable_variables
(	variables
)regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
-trainable_variables
.	variables
/regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
.
20
31"
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
6trainable_variables
7	variables
8regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
<trainable_variables
=	variables
>regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Atrainable_variables
B	variables
Cregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Etrainable_variables
F	variables
Gregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Jtrainable_variables
K	variables
Lregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Otrainable_variables
P	variables
Qregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@@2dense_3/kernel
:@2dense_3/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
Utrainable_variables
V	variables
Wregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Ztrainable_variables
[	variables
\regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
.
_0
`1"
trackable_list_wrapper
<
_0
`1
a2
b3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
ctrainable_variables
d	variables
eregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
.
h0
i1"
trackable_list_wrapper
<
h0
i1
j2
k3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
ltrainable_variables
m	variables
nregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
ptrainable_variables
q	variables
rregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
ttrainable_variables
u	variables
vregularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :T2dense_4/kernel
:2dense_4/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
ztrainable_variables
{	variables
|regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adamax/iter
 : (2training/Adamax/beta_1
 : (2training/Adamax/beta_2
: (2training/Adamax/decay
':% (2training/Adamax/learning_rate
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
X
0
 1
42
53
a4
b5
j6
k7"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?trainable_variables
?	variables
?regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7:52+training/Adamax/batch_normalization/gamma/m
6:42*training/Adamax/batch_normalization/beta/m
0:.@2 training/Adamax/dense_2/kernel/m
*:(@2training/Adamax/dense_2/bias/m
.:,2training/Adamax/dense/kernel/m
(:&2training/Adamax/dense/bias/m
9:7@2-training/Adamax/batch_normalization_2/gamma/m
8:6@2,training/Adamax/batch_normalization_2/beta/m
0:.2 training/Adamax/dense_1/kernel/m
*:(2training/Adamax/dense_1/bias/m
0:.@@2 training/Adamax/dense_3/kernel/m
*:(@2training/Adamax/dense_3/bias/m
9:7@2-training/Adamax/batch_normalization_3/gamma/m
8:6@2,training/Adamax/batch_normalization_3/beta/m
9:72-training/Adamax/batch_normalization_1/gamma/m
8:62,training/Adamax/batch_normalization_1/beta/m
0:.T2 training/Adamax/dense_4/kernel/m
*:(2training/Adamax/dense_4/bias/m
7:52+training/Adamax/batch_normalization/gamma/v
6:42*training/Adamax/batch_normalization/beta/v
0:.@2 training/Adamax/dense_2/kernel/v
*:(@2training/Adamax/dense_2/bias/v
.:,2training/Adamax/dense/kernel/v
(:&2training/Adamax/dense/bias/v
9:7@2-training/Adamax/batch_normalization_2/gamma/v
8:6@2,training/Adamax/batch_normalization_2/beta/v
0:.2 training/Adamax/dense_1/kernel/v
*:(2training/Adamax/dense_1/bias/v
0:.@@2 training/Adamax/dense_3/kernel/v
*:(@2training/Adamax/dense_3/bias/v
9:7@2-training/Adamax/batch_normalization_3/gamma/v
8:6@2,training/Adamax/batch_normalization_3/beta/v
9:72-training/Adamax/batch_normalization_1/gamma/v
8:62,training/Adamax/batch_normalization_1/beta/v
0:.T2 training/Adamax/dense_4/kernel/v
*:(2training/Adamax/dense_4/bias/v
?2?
%__inference_model_layer_call_fn_16393
%__inference_model_layer_call_fn_15765
%__inference_model_layer_call_fn_15653
%__inference_model_layer_call_fn_16424?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_14410?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
@__inference_model_layer_call_and_return_conditional_losses_16211
@__inference_model_layer_call_and_return_conditional_losses_15542
@__inference_model_layer_call_and_return_conditional_losses_15462
@__inference_model_layer_call_and_return_conditional_losses_16362?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
3__inference_batch_normalization_layer_call_fn_16535
3__inference_batch_normalization_layer_call_fn_16544?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16503
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16526?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_16577?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_16570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_16610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_16603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_batch_normalization_2_layer_call_fn_16721
5__inference_batch_normalization_2_layer_call_fn_16730?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16689
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16712?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_16763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_16756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_tf_op_layer_mul_layer_call_fn_16775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_16769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_re_lu_layer_call_fn_16785?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_re_lu_layer_call_and_return_conditional_losses_16780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_tf_op_layer_mul_1_layer_call_fn_16797?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_16791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_tf_op_layer_add_layer_call_fn_16809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_16803?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_16842?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_16835?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_tf_op_layer_add_1_layer_call_fn_16854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_16848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_batch_normalization_3_layer_call_fn_16974
5__inference_batch_normalization_3_layer_call_fn_16965?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16956
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16933?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_1_layer_call_fn_17094
5__inference_batch_normalization_1_layer_call_fn_17085?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17053
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17076?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_re_lu_1_layer_call_fn_17104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_17117?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_17111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_17135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_17128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_17148?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_17163?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_17178?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_17193?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
2B0
#__inference_signature_wrapper_15986input_1
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16503b 3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16526b 3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
1__inference_tf_op_layer_mul_1_layer_call_fn_16797vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
5__inference_batch_normalization_1_layer_call_fn_17094Ukhji3?0
)?&
 ?
inputs?????????
p 
? "???????????
/__inference_tf_op_layer_add_layer_call_fn_16809vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16933bab_`3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
B__inference_dense_3_layer_call_and_return_conditional_losses_16835\ST/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? t
%__inference_re_lu_layer_call_fn_16785K/?,
%?"
 ?
inputs?????????@
? "??????????@v
'__inference_re_lu_1_layer_call_fn_17104K/?,
%?"
 ?
inputs?????????@
? "??????????@?
 __inference__wrapped_model_14410? %&5243+,:;STb_a`khjixy0?-
&?#
!?
input_1?????????
? "1?.
,
dense_4!?
dense_4??????????
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16956bb_a`3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
@__inference_model_layer_call_and_return_conditional_losses_15462} %&4523+,:;STab_`jkhixy8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
+__inference_concatenate_layer_call_fn_17117vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????@
? "??????????T?
1__inference_tf_op_layer_add_1_layer_call_fn_16854vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_16211| %&4523+,:;STab_`jkhixy7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
#__inference_signature_wrapper_15986? %&5243+,:;STb_a`khjixy;?8
? 
1?.
,
input_1!?
input_1?????????"1?.
,
dense_4!?
dense_4??????????
%__inference_model_layer_call_fn_16424o %&5243+,:;STb_a`khjixy7?4
-?*
 ?
inputs?????????
p 

 
? "??????????z
'__inference_dense_1_layer_call_fn_16763O:;/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_tf_op_layer_mul_1_layer_call_and_return_conditional_losses_16791?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
5__inference_batch_normalization_3_layer_call_fn_16974Ub_a`3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
5__inference_batch_normalization_2_layer_call_fn_16721U45233?0
)?&
 ?
inputs?????????@
p
? "??????????@?
B__inference_dense_1_layer_call_and_return_conditional_losses_16756\:;/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? :
__inference_loss_fn_1_17163+?

? 
? "? ?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16689b45233?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? x
%__inference_dense_layer_call_fn_16610O+,/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_16362| %&5243+,:;STb_a`khjixy7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
L__inference_tf_op_layer_add_1_layer_call_and_return_conditional_losses_16848?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
B__inference_dense_2_layer_call_and_return_conditional_losses_16570\%&/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? :
__inference_loss_fn_0_17148%?

? 
? "? z
'__inference_dense_4_layer_call_fn_17135Oxy/?,
%?"
 ?
inputs?????????T
? "???????????
5__inference_batch_normalization_3_layer_call_fn_16965Uab_`3?0
)?&
 ?
inputs?????????@
p
? "??????????@z
'__inference_dense_2_layer_call_fn_16577O%&/?,
%?"
 ?
inputs?????????
? "??????????@?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16712b52433?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? z
'__inference_dense_3_layer_call_fn_16842OST/?,
%?"
 ?
inputs?????????@
? "??????????@?
3__inference_batch_normalization_layer_call_fn_16535U 3?0
)?&
 ?
inputs?????????
p
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_15542} %&5243+,:;STb_a`khjixy8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_16803?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
@__inference_dense_layer_call_and_return_conditional_losses_16603\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
3__inference_batch_normalization_layer_call_fn_16544U 3?0
)?&
 ?
inputs?????????
p 
? "??????????:
__inference_loss_fn_3_17193S?

? 
? "? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17053bjkhi3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? :
__inference_loss_fn_2_17178:?

? 
? "? ?
/__inference_tf_op_layer_mul_layer_call_fn_16775vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
@__inference_re_lu_layer_call_and_return_conditional_losses_16780X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_16769?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_15765p %&5243+,:;STb_a`khjixy8?5
.?+
!?
input_1?????????
p 

 
? "???????????
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17099X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17076bkhji3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_16393o %&4523+,:;STab_`jkhixy7?4
-?*
 ?
inputs?????????
p

 
? "???????????
%__inference_model_layer_call_fn_15653p %&4523+,:;STab_`jkhixy8?5
.?+
!?
input_1?????????
p

 
? "???????????
5__inference_batch_normalization_2_layer_call_fn_16730U52433?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
B__inference_dense_4_layer_call_and_return_conditional_losses_17128\xy/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????
? ?
5__inference_batch_normalization_1_layer_call_fn_17085Ujkhi3?0
)?&
 ?
inputs?????????
p
? "???????????
F__inference_concatenate_layer_call_and_return_conditional_losses_17111?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????@
? "%?"
?
0?????????T
? 