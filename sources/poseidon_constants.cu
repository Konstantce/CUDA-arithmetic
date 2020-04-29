#include "cuda_structs.h"
#include "poseidon.h"


DEVICE_FUNC CONST_MEMORY uint256_g ARK[NUM_ROUNDS][m] = {
{
	{ 0x033140bc, 0x714c6181, 0x724c7198, 0x6f130067, 0xe97ce9f1, 0xe2dcfc8c, 0x124a22df, 0x04e656d5 },
	{ 0xcb6243ae, 0x4dae2074, 0xb99e3f78, 0x259ccedd, 0x08c19fa1, 0x3c246819, 0x01dac744, 0x0c9f250f },
	{ 0x5c31c5cf, 0x4c3e26ca, 0x51f4deb9, 0x5fc1f12b, 0x793a7420, 0x391fdddc, 0x054de1bc, 0x0816fde0 }
},

{
	{ 0xc8441a40, 0xfcae45a2, 0x1cb5b5a9, 0x9927f6a4, 0x8f4075d6, 0xcb25e740, 0x3e53a37d, 0x2e2220df },
	{ 0x8868c442, 0x029f4902, 0x21ef0929, 0x163e8a8d, 0xefc691fb, 0x3a80aa8f, 0x3c06111e, 0x115d4269 },
	{ 0x823c6c69, 0xf410c042, 0x1ce99856, 0xadc4e2e9, 0xe849eaad, 0x994e68c8, 0x281a4c9c, 0x23096fa6 }
},

{
	{ 0x698add70, 0x73b65055, 0xdc20dc9a, 0x5f21e4b7, 0xe51bf664, 0xcde2fdb0, 0x8d90a2c9, 0x1b4e6b42 },
	{ 0x3f85c341, 0xc969a7cc, 0xa1ac9385, 0x728f209e, 0x69755163, 0xfc64d002, 0xd1389725, 0x159f72bd },
	{ 0x5613a47f, 0x66debea4, 0x2a526beb, 0xe15fbc95, 0x6d41ae2e, 0x63be4f2d, 0x70b91c7f, 0x0d080d2f }
},

{
	{ 0xa54f6ea6, 0x2d4e866b, 0x851cf5ae, 0x37d0c956, 0xf98c660e, 0x1653aa72, 0x611461ab, 0x084fc92d },
	{ 0xdfbc10d1, 0xe23f1759, 0x287c669e, 0x7c479cb2, 0x94d44c60, 0x91a38948, 0x1d5826c8, 0x1148b56c },
	{ 0xa92d5cb9, 0xa82eecbc, 0xf39dd482, 0xd26e5301, 0xa4020cd9, 0x80cc441a, 0x75e65e28, 0x226bb9e3 }
},

{
	{ 0xd7bf0f21, 0x616a12b8, 0x728112f8, 0xfddc5280, 0xd7b0d227, 0x59e2cc5b, 0x9c7af5d0, 0x053916a1 },
	{ 0x3c709413, 0x019310b1, 0xa61e5295, 0xa183c63a, 0x4ce35052, 0x73f2291b, 0x5d7342cb, 0x12e302d8 },
	{ 0x47cc587a, 0x9c139978, 0x16589ba2, 0x01e83fc4, 0x4a3726b0, 0x432216cf, 0xeffff2a9, 0x0d71af76 }
},

{
	{ 0xb5233be0, 0x48529924, 0x204c5c59, 0x95418379, 0xffa9b92c, 0x99eed14f, 0x9d54821c, 0x23a71de8 },
	{ 0x05063c9a, 0x1d87da03, 0xe8015052, 0x307e4f89, 0xa9d2ad2f, 0xd2b41747, 0x412da2ce, 0x0242e2ac },
	{ 0xfeda92ca, 0xe1e757a8, 0x20083c70, 0xe7fc7b1c, 0x3c798dc5, 0x29626a6c, 0x44805dd5, 0x269be4e9 }
},

{
	{ 0x102fcc88, 0x8ba34710, 0x924bd76e, 0x0248655f, 0xbe6a0315, 0xa155434a, 0x57cfb326, 0x26fbbdaf },
	{ 0x6d650f3f, 0xd9f59e1a, 0xfddd047c, 0x25daf2b4, 0xa262f95b, 0xeaab57e2, 0x0983a859, 0x2c0c427f },
	{ 0xb380db72, 0x28e3d688, 0xdf3711b4, 0xc5581bfd, 0xaea70edd, 0xe70dbc2f, 0xaefc8036, 0x28d8cbcd }
},

{
	{ 0xc1f94c62, 0x70a117da, 0x1080df08, 0x81db5ad6, 0x84965801, 0x6781c37f, 0x79efbc0c, 0x135fe4a4 },
	{ 0x5e63edd3, 0x9e413898, 0x255849cf, 0x53ed2300, 0x329cb88e, 0x5301c468, 0x7d79cfc8, 0x0e772632 },
	{ 0x2eca852b, 0x1c1d8754, 0x9fbfea62, 0x2d70ff28, 0x66e7263e, 0xd218f394, 0x522b7cc9, 0x0be1ab71 }
},

{
	{ 0x33c02991, 0xa4760885, 0x738f2469, 0xb1905190, 0x45075b3d, 0x0a1e09b8, 0xfe38fdf4, 0x01c3be78 },
	{ 0xf19cea55, 0x06fb64f0, 0x20e086da, 0x890cbcca, 0x67e551ed, 0xfa6f2d42, 0xcd329f34, 0x1d437dd6 },
	{ 0x6839a96f, 0x6ed3cb8d, 0xffac8c83, 0x9f8ee60d, 0xdc4e8ca2, 0x12673e43, 0x04b9ea79, 0x21bf2807 }
},

{
	{ 0x5f25edde, 0x50d27294, 0x22c05a17, 0xeec8ff50, 0x95b7be99, 0x9741db01, 0x7af9d6c0, 0x2080c68c },
	{ 0x515cc4ef, 0xff306166, 0xfa92e391, 0x64383964, 0x4779d1c1, 0x62ad5140, 0xbedf938e, 0x047945ad },
	{ 0x6a9a4046, 0x7394ec45, 0x84c7527b, 0xb9cbe06d, 0xeba630a3, 0x6fed87db, 0x6b954734, 0x1b38a5ee }
},

{
	{ 0xaded26dc, 0x1b91d1ac, 0x94516b86, 0xbdaab615, 0xece8dd8e, 0xeb492629, 0xec563c32, 0x035a637a },
	{ 0x6c3da4d1, 0x180cca27, 0x53d0d793, 0x9e3a6390, 0x70bc2ce6, 0x46a12a83, 0x092811ff, 0x1492be2b },
	{ 0x83174be2, 0x2035de38, 0x684ea747, 0xd1943ae3, 0xbf0557d8, 0x44fc0a9d, 0xe22bd42a, 0x0f8a0efe }
},

{
	{ 0x6b31fd9d, 0x3b3ee2d1, 0x6de8dced, 0x38fadf2a, 0x6bb88cc1, 0x6fcc9390, 0x7bff86f0, 0x104b90d1 },
	{ 0xf516d9c6, 0xa89e571d, 0x19378467, 0x4685d895, 0x915d6461, 0xab820f29, 0xf0375a6b, 0x10105dd9 },
	{ 0x0582860e, 0x4a0c5d34, 0xb7af2371, 0x12146ded, 0x0dc68477, 0xd83a6390, 0xbab20e54, 0x2f9ad298 }
},

{
	{ 0x7b2296ee, 0x9b7a4252, 0x757a42b3, 0x2dcb8840, 0xcf8639ed, 0x43723d77, 0x254339cb, 0x09dc7674 },
	{ 0xa3c5cb00, 0xc3126793, 0x40197d0c, 0xa4c330fe, 0x26241a12, 0xb6c2f86b, 0x4691e2e6, 0x298448c6 },
	{ 0x3190757a, 0x9a59094b, 0x6ea52baf, 0x103b392d, 0x0b7fe281, 0xe5ff893c, 0x851c657d, 0x12285b17 }
},

{
	{ 0x223e6471, 0x7939d743, 0x6e39563f, 0xdc8a4ad7, 0x7181bb1f, 0xf910701c, 0xc0f7dae1, 0x1619f600 },
	{ 0x19b76b4f, 0xcf4bb10a, 0xe71e0eca, 0xe5ddc9d4, 0xcae90bb6, 0x7fcd27f1, 0xa9a11cf2, 0x287cf7da },
	{ 0xf4dbf346, 0x7f7e92e9, 0xbd32cf6c, 0x5d9668c1, 0x1b289d96, 0xe9279d30, 0xb30d531e, 0x24ecb0c8 }
},

{
	{ 0x3e6a8306, 0x20c61a85, 0x0acd364e, 0x7e7ffe9c, 0xeaee0ca1, 0xca20e46b, 0x43b5eb17, 0x035c55cb },
	{ 0x1bf60b10, 0x97ccd018, 0x1ec7f8e4, 0x752bf3e8, 0x210f7156, 0xc8ca1e12, 0x669f6410, 0x03668228 },
	{ 0x5da98124, 0x97359101, 0x47ad74e8, 0x74d2c842, 0x109dc447, 0xd50722ee, 0x15d816c4, 0x14107160 }
},

{
	{ 0x28456a45, 0xa325ba80, 0x691c9b49, 0x577244fd, 0x4f4db9d0, 0x1c593a3f, 0x4eec5ac4, 0x0232b5fa },
	{ 0xff434b8c, 0xc8bc5d02, 0x5a75d817, 0x8543d22c, 0xf466bc82, 0xa8675bb6, 0x0f57ddde, 0x2971b913 },
	{ 0xef19bc1e, 0x6659a09e, 0xf605e5ac, 0x46dd459f, 0xab1a010d, 0x09792ef1, 0xa53cfcbe, 0x02e078fd }
},

{
	{ 0xe0f05d5f, 0x42d05ca2, 0xa7a39394, 0xe5bbc97a, 0x63cc8c2b, 0x4a833303, 0xa3dd593f, 0x1febb93c },
	{ 0xa2390e68, 0xc90a4823, 0x319a29b7, 0x281f78c1, 0x13bd463c, 0xc31cf370, 0x6defd080, 0x01f68fea },
	{ 0xb4f3075b, 0xbd796738, 0x784d57cd, 0x3b66539a, 0x92d8f1df, 0xd8dc7027, 0xe34d36ba, 0x2ac72f82 }
},

{
	{ 0xda4ecc14, 0x3c88555d, 0x208270fd, 0xf2d43b41, 0xb5162d1d, 0xda309ffb, 0xf6812261, 0x1769d4b0 },
	{ 0x3f67501a, 0xc841b737, 0x3ff45e8c, 0x9008b4a9, 0xb39a1837, 0xadf7588a, 0x0d0e4b52, 0x12e41ffc },
	{ 0x2754a17e, 0x7cfcb36a, 0x2b9f4cbb, 0x65237fb6, 0xc46e4ae6, 0x352f433b, 0xc760a976, 0x0c689cf2 }
},

{
	{ 0x44a172c5, 0x801f12f3, 0x4ccb0de6, 0xcef31ebc, 0xc7beb6f5, 0xb94b0556, 0x18aff9d9, 0x00e0fb13 },
	{ 0xd7c7d295, 0x4aaa7b1e, 0x2c4c8083, 0xc581e992, 0xbbba7842, 0x0752cb03, 0xd1203628, 0x174f20c1 },
	{ 0x5e8a7c46, 0xe2231814, 0xa77af978, 0xd41295ed, 0x48213cbe, 0x14dafa8b, 0xaac3b2f6, 0x0c6957c0 }
},

{
	{ 0x7c89143e, 0x3f96ef3b, 0x20db0225, 0x8081fa93, 0xe2103f52, 0x152139b4, 0x13654a5f, 0x089a8e35 },
	{ 0x1d29f365, 0xbdbc9c1d, 0x1957c4bc, 0x1a9050e4, 0x3f71b9a0, 0x6705a723, 0x7d86c2d2, 0x16d87439 },
	{ 0xb7a64c91, 0x3b4e4cf8, 0x66a9d571, 0x5751f22d, 0xe3928706, 0xf255f462, 0x93785110, 0x0701c936 }
},

{
	{ 0xc843c0b9, 0xcd92dfd6, 0x23c5de37, 0x127382bb, 0xa288d1f1, 0x72602f9e, 0x55b84a60, 0x07e98b16 },
	{ 0x0fb89848, 0x29200ff6, 0xac5eeac5, 0x335c0490, 0xce3eb843, 0xdbdb5a94, 0xd403cc3e, 0x17ef3fb9 },
	{ 0xc1454dcb, 0x4fa0c2d7, 0x9c1b37e0, 0x50744fae, 0xb00dfdb4, 0x58ea7b16, 0x736aa6a6, 0x201a3faa }
},

{
	{ 0xaa929c04, 0x9599273e, 0x718e6f5c, 0x60aebe7c, 0x38dc7b72, 0x0a2de6c6, 0xdc2492cd, 0x084b919e },
	{ 0xeba3a286, 0x26c3ac55, 0xd1f7ced8, 0xba71dd8c, 0x840f8fd3, 0x17f552f1, 0xf4bfe57c, 0x10d0828d },
	{ 0xee56fecf, 0x40f66897, 0xab3bf866, 0xc4ec9b74, 0x3074cc35, 0x32f4c3e9, 0x8cb8abec, 0x2a8ae61e }
},

{
	{ 0xef15899f, 0xb98a0f87, 0xc24e3d6a, 0xaa0ccd9b, 0xb20e70e0, 0xd090e523, 0xba37b7fd, 0x1e5e397e },
	{ 0x80702f3a, 0x4be7e36c, 0x90f6e28b, 0x69f3b5c3, 0xb0d5be7c, 0xc80aecf4, 0xef74b46c, 0x2f670554 },
	{ 0x6aadab50, 0xd614d5cd, 0xcedfe45f, 0xa5a83191, 0xafe628e8, 0x038265ec, 0x7c9e8feb, 0x064b2612 }
},

{
	{ 0xf356509a, 0xdda2ccca, 0x3eef2be0, 0xce9fab57, 0xd522552d, 0x76a0557d, 0xbc07cb49, 0x126e0202 },
	{ 0x0955a7c8, 0xf202401b, 0x6c763e03, 0x0c99b513, 0x8e977622, 0x6da4c318, 0x006c06e8, 0x157d5260 },
	{ 0xee1e170f, 0x5d111d20, 0x026ef6e2, 0xf15bf1ff, 0x617f035b, 0xff63b10c, 0xf68afde6, 0x017933f7 }
},

{
	{ 0x44fe8257, 0x6a6ac271, 0xdb662eff, 0x34b72680, 0xd30c0500, 0x7eb529ab, 0x01e97762, 0x27498738 },
	{ 0x1aa06bc3, 0x3d2db03b, 0x81e5dce5, 0xbe196b83, 0xc07b9bfd, 0xd6c7e235, 0xe28a8332, 0x0dd21848 },
	{ 0x1fdf6d9a, 0x70374edc, 0x34ca3372, 0x4fa3e2df, 0x134a4f74, 0x0077fb36, 0x69e237fa, 0x0fe4eb5f }
},

{
	{ 0xcb2cbc07, 0x7a9a19a5, 0x2f60a383, 0x652ba1ca, 0x4c51a12d, 0x02e703c2, 0x3b5553ec, 0x0e30a792 },
	{ 0xf94e5abd, 0xb424e1e2, 0x4b48560b, 0xd8485575, 0xa331f182, 0x0cdc9cd0, 0xe5a3aa88, 0x2cf394ee },
	{ 0xf9035a50, 0x029c2e03, 0xb5c0d8a0, 0x4e244066, 0xb40de84a, 0x5e0390c5, 0xde779db0, 0x2284dd16 }
},

{
	{ 0x5612335d, 0x3c6ee698, 0x526b3232, 0xc22ac469, 0xf2d04d00, 0x528fdc8d, 0x986d0961, 0x0d148fb9 },
	{ 0x4ac2284d, 0xa91f511f, 0x7f50688d, 0xb759b29f, 0xa910e408, 0x9f106278, 0x19b2e2b5, 0x15d91563 },
	{ 0x4239c647, 0x11f45e30, 0x9e9fea86, 0xf3c13a24, 0x69c51696, 0x3e0d8fac, 0xef482b79, 0x0c629186 }
},

{
	{ 0x0ced5fb5, 0xc456df2d, 0x2fdad0e9, 0xbb03df60, 0x2166b892, 0x9c19baf2, 0x9fd92d1a, 0x08e24522 },
	{ 0x8afe9fd1, 0xe53016b3, 0x2e317558, 0xeb09556d, 0xf17e8594, 0x6f8ba35a, 0xe257cde7, 0x2e1f2477 },
	{ 0xd29711bf, 0x242a2b60, 0x26076a8e, 0x15eff6dd, 0x1e98a954, 0x03dd77e2, 0x88580aa3, 0x078761da }
},

{
	{ 0xda051c16, 0x4598957b, 0x01c4a91d, 0x7edf114f, 0x049ba8a7, 0x8f1aa27c, 0x85ce82b4, 0x1663f9be },
	{ 0x8122de8b, 0x48e7f6b8, 0x73bb2645, 0xd9faa883, 0x9052da1c, 0x6788cf63, 0xba85b903, 0x0411f0ee },
	{ 0xe6eb6b3d, 0xbebf89c8, 0xc0de58ca, 0x4e25cccb, 0xa2870f34, 0xd8976d8c, 0xa5eed3cf, 0x00708e2b }
},

{
	{ 0xc3368202, 0x5f3d09ed, 0xdd00283c, 0x8bf5d0ae, 0x19242fd6, 0x2a732089, 0xa190dbb2, 0x07852803 },
	{ 0x0e08d9fd, 0x6f8e0d25, 0x2f6bbe25, 0x541e1ecb, 0x5211103a, 0x1261aabe, 0x941f1349, 0x010ec58d },
	{ 0x61bf76aa, 0x94f0cc85, 0x9e6baf87, 0x6bb7e9a8, 0xaf6e918c, 0x6bbfce8a, 0xba556766, 0x193331ca }
},

{
	{ 0xbbfe9cef, 0x8e38fb2c, 0x950d89f6, 0x32982d29, 0x7e0ffd33, 0x9226bdbe, 0x3ef630a3, 0x2a323754 },
	{ 0xbdfdd699, 0xec9bd947, 0xd61d2ac5, 0xfb5b4202, 0x0380ed2b, 0xa2cb7efa, 0xc4663da9, 0x2fcf1912 },
	{ 0xa4f46ecf, 0x4fd33e73, 0xd560be77, 0x9ff75df2, 0xb2c96f67, 0xd5192dc9, 0x37b4a87f, 0x134d57ee }
},

{
	{ 0x7edf6db4, 0xd24c3216, 0x46643750, 0x22f577a7, 0xfccd61d4, 0xcfd6237e, 0x8f0ee1b5, 0x29bbc16a },
	{ 0xeee93e2e, 0x45141429, 0x8dea08b1, 0xd53f7939, 0xdfbd8229, 0x75aa6f28, 0x638c01bc, 0x21455555 },
	{ 0x9eb8a187, 0x12656563, 0x1b79a50a, 0xfba8e204, 0xa2969beb, 0x67480da4, 0x2817802b, 0x020741ad }
},

{
	{ 0x20fba23f, 0x65505188, 0xdcbc390a, 0x3740f3b0, 0xb639117f, 0xc2056f81, 0x87cfee49, 0x27f01cc6 },
	{ 0xcf6733fc, 0x4aead79d, 0x06c41d43, 0x66dd4a42, 0xe4a10224, 0x7441da1d, 0x6cc3f03e, 0x0d6855c2 },
	{ 0x9744998c, 0xc4a66f86, 0xce5a9056, 0xa1b1bf1b, 0x7c37522a, 0x4cbe29c7, 0xfb652ddf, 0x061a338e }
},

{
	{ 0x0f93b015, 0x170c189b, 0x98995631, 0x8eff913a, 0xe85394ac, 0xbef86c74, 0x711471e2, 0x08047eeb },
	{ 0x2d9f02fb, 0x78c4e128, 0x3f19580d, 0x83c4f40d, 0xe9eb9db2, 0x6d127cff, 0x8f3fd017, 0x2564342e },
	{ 0x6c284c00, 0x72b3389d, 0x109b6ec2, 0x77a90b06, 0x1eaf35c6, 0xd61c62ae, 0x20a6ce4d, 0x2a2edec7 }
},

{
	{ 0x7f4d1756, 0xd200c694, 0xd73c2cbf, 0x6babd706, 0x0d160ca1, 0x1e56dc31, 0x505774e6, 0x0e4976a1 },
	{ 0x01278d85, 0x66bc9008, 0x31532b82, 0x7e2753b2, 0xc531c382, 0xf6b28881, 0x0225d4c6, 0x1cbedda1 },
	{ 0x7e86f33c, 0xf1d0f569, 0x17b9f851, 0x0906f341, 0x3f0e5c0c, 0xb0c959bb, 0xfd8dacc6, 0x14e3674d }
},

{
	{ 0x1411e7c1, 0x34118b81, 0x5f906f7e, 0xba1e5786, 0x0dc5bce8, 0x142a94c3, 0x11d70a3d, 0x1dbb1f2d },
	{ 0xc9817b96, 0xa3dbd3b5, 0x8a69c8a0, 0x4311c12d, 0xb419d9b9, 0xf3efaa8b, 0xe1709b26, 0x2b51bc6d },
	{ 0x68afe439, 0xe3c7b438, 0x55dcc0de, 0x5fc6af9e, 0xd69b0cc5, 0xac2e7da7, 0x69e572f2, 0x2ccd7f15 }
},

{
	{ 0xad5b36d0, 0xf0b3b9da, 0xed333c45, 0x39c90544, 0x9f45ca11, 0x5f471047, 0xc76493c8, 0x2c8f5d0e },
	{ 0x5de51773, 0xbcc82fa8, 0xc65c9568, 0x1258b2a1, 0x626394f5, 0xe324bb44, 0x79701bc1, 0x14b17971 },
	{ 0xabb17805, 0xba3d2126, 0x81216db6, 0xf2b9f849, 0x3a2717e3, 0x3309fe0c, 0xd113218f, 0x054a47f1 }
},

{
	{ 0x8daf839e, 0x498cb6a6, 0x57832915, 0x9584cac8, 0x98bc0fba, 0x36e6ef1c, 0xde7083b8, 0x0d31e846 },
	{ 0x9f499116, 0xc922f3cf, 0xb4d90a08, 0xa952cf2a, 0x180601d1, 0x12a8ad9e, 0xbfc5920f, 0x0194ddf3 },
	{ 0x0d0de7de, 0x13d2b2b6, 0x9d3d0b92, 0x5a5a92fa, 0xf26872d7, 0xd6deaa51, 0xb98d51b7, 0x1c3a357e }
},

{
	{ 0xc785b501, 0x4830d3c7, 0x5d844986, 0xb0e67fa1, 0xd7e68489, 0x6a9f000c, 0x5b7ef797, 0x07c2cea6 },
	{ 0xeb9b32bb, 0x59a054e3, 0xe3dfd3c1, 0x81daad0d, 0x90dff1e0, 0x5168f800, 0xd80444fa, 0x1e1bb7a5 },
	{ 0x6f48ccef, 0x53bd3bf7, 0xe293d798, 0x912309cd, 0xdc69bf96, 0xc0367d6c, 0x8bb6e3fb, 0x25fb609d }
},

{
	{ 0x8f11ab10, 0x0d0c567d, 0xd6919998, 0x486a497a, 0xd691f757, 0xa6dba1fa, 0x770023ab, 0x07ac93ec },
	{ 0x80c24892, 0x5d8b757a, 0xb7d8b59a, 0x590c3a0e, 0x913b5143, 0x4628a1a9, 0xd3b1992a, 0x009cfffe },
	{ 0x3d34812a, 0xfc571f2f, 0xd24214fa, 0x8eec7c56, 0xe7e26169, 0xda0159b6, 0x8d97e1ee, 0x265c72ad }
},

{
	{ 0xa8022426, 0x983e62d6, 0x65084252, 0x627c0d5e, 0x0e3561bd, 0x63a7e7d9, 0x4e8a9dd5, 0x20a48e48 },
	{ 0xae57d144, 0x0283bd6e, 0x963d14b4, 0xfe589641, 0x78daa581, 0xcba1555a, 0xba91c722, 0x29b498db },
	{ 0x9f674ec5, 0xce66b5dc, 0xaf9e697c, 0x0e685275, 0xec316e45, 0x1864cfc4, 0x7f76a9d1, 0x17079db3 }
},

{
	{ 0x71537bde, 0xff6657b1, 0xf2b7491f, 0x33c7757b, 0xa8dc8d07, 0xf4bf048c, 0x45c73104, 0x0f31388e },
	{ 0xacb81f08, 0xc1caa67c, 0xec949588, 0x1ef5492c, 0xe432ca98, 0x515f7f89, 0xa73b6395, 0x1994b481 },
	{ 0xadf6a53d, 0x6f2b0e53, 0xba67af2a, 0x19c561d9, 0x5a473d89, 0xb9b9aac8, 0x0ea08929, 0x203a38aa }
},

{
	{ 0x5981dff4, 0x153f22a9, 0xa28d0535, 0x844ae074, 0x966e041f, 0xa2c71806, 0x60058620, 0x1bd0b4e9 },
	{ 0xf5f7fffc, 0x9c897812, 0xbde36d08, 0xe7c391f8, 0x6faeba51, 0x723ce807, 0x4c96be1c, 0x235e59ea },
	{ 0x8f577b2c, 0x62d632b5, 0x34fd5550, 0x0596cd15, 0x3615bd7b, 0x4f02e687, 0x66cf2102, 0x0b354fcf }
},

{
	{ 0xf6fd4f2c, 0xd513308b, 0x32a9174b, 0x4cde5350, 0x73ac197b, 0xc8dfc3f1, 0xb40307c4, 0x2fbe6929 },
	{ 0xc056ab70, 0x241293b3, 0xabd4be9a, 0x4dd8a70d, 0x1f1fdc57, 0x17973c5d, 0x47464438, 0x152fb155 },
	{ 0x11922d3e, 0x39e7ddbe, 0x82e45ff4, 0x067d71ce, 0xf0b896d6, 0x4e490dbe, 0x5c213017, 0x2785d104 }
},

{
	{ 0x8f65f9b1, 0x044fa010, 0x1bb582c7, 0x2daf1863, 0xf6f7edba, 0x35556df2, 0xb330b5df, 0x2643cf5a },
	{ 0x95010996, 0x20bb6411, 0x61360452, 0x9e52f28b, 0xe4d9409b, 0x99647958, 0x0394caa8, 0x21ad2c3d },
	{ 0x49a51b15, 0x3da991d4, 0xa374cab2, 0x07be1c43, 0x39d892ef, 0xb388bcdb, 0xb2fc4e0e, 0x18ecaa16 }
},

{
	{ 0x36c1b5ee, 0x8977fb23, 0x376b2bef, 0xab33c531, 0x229e4391, 0x149b2e26, 0xbed1375c, 0x244be253 },
	{ 0xffc89f61, 0x304b289a, 0x94822b40, 0xb04b26e9, 0x0f30096d, 0xe4714f32, 0x321a0dcc, 0x2dcd8906 },
	{ 0x4738dab2, 0x775a4eef, 0x67113733, 0xd3d70518, 0xb6866b76, 0xc96ab886, 0xe0dea183, 0x0f935310 }
},

{
	{ 0xb77371fd, 0x73d086e9, 0xc1d5e4e5, 0x8af23240, 0x4ad7c282, 0x727f6a27, 0x484830ed, 0x0165f6f8 },
	{ 0x34965113, 0xa50fd2c5, 0x196db575, 0x882d9c94, 0xae2d0d32, 0xc8336874, 0xbbf3805e, 0x0baa5046 },
	{ 0x66f53796, 0x6d0740de, 0x56df8d81, 0xcd531b54, 0x3a1ad86b, 0xac679894, 0xcc513f54, 0x0a90f460 }
},

{
	{ 0x3802900f, 0x89742ce2, 0xd603cc8e, 0xf6754b66, 0xc69a0630, 0x6eb23a58, 0x43910f6f, 0x23fc1276 },
	{ 0xa50edc9e, 0x87a3554b, 0x563de46d, 0x87db3d58, 0x37992dce, 0x14069aa0, 0xfc74ba92, 0x294a598e },
	{ 0xa712ed13, 0x30fa3eab, 0xeb5aefd6, 0xa35d3344, 0x490d9678, 0x8192883e, 0x9ba51b94, 0x213b8c15 }
},

{
	{ 0xd313f0e4, 0x785ce8a9, 0x50e0e58a, 0x39a9c26b, 0x9182ebb9, 0x373c0c51, 0xd368eb7f, 0x25157d35 },
	{ 0x71e49119, 0xc4090531, 0x7639b8b2, 0x76c7b8e5, 0x61082107, 0xe6ea8b40, 0x6603c7d4, 0x02aaa8ce },
	{ 0x30105171, 0x50367d02, 0x3e2bbec5, 0xc858c624, 0xd75ce27a, 0x8f374ee9, 0x6226191c, 0x09f10f61 }
},

{
	{ 0xc6bebf0f, 0x0ffbd00b, 0x84650702, 0x177a6e2f, 0x1eb576b8, 0x7020a2c5, 0x4f5ba226, 0x29537062 },
	{ 0x81834468, 0x8ab66553, 0x1711a385, 0x60ce6375, 0x997645c7, 0x2ea07208, 0xb3058a8c, 0x211c60cc },
	{ 0x6ecf6636, 0xb4d7e313, 0x00132486, 0x1f979211, 0x6fbf772b, 0x9033325a, 0x806cc8b5, 0x0a1252c3 }
},

{
	{ 0xacf33127, 0x4506dc04, 0x3a5bbb7b, 0x79fcebf6, 0x1f0ab85e, 0x1d6277a6, 0x7f4d3c00, 0x2b4975ad },
	{ 0xf97285ea, 0x5878d8cd, 0xf3169233, 0xc893f930, 0xcf4094da, 0x0edcf70e, 0xa39c8bd7, 0x2c19f5f5 },
	{ 0x916f4cda, 0x1af80d8e, 0x5e8c4738, 0x71e1515c, 0xf6ea1a24, 0xc9806d78, 0x6f4655e9, 0x0313c9ba }
},

{
	{ 0x3bc96e00, 0xfd24c030, 0x0bb4f10f, 0x3adfdaa2, 0xed208677, 0xd4ca2fc8, 0x8092ff34, 0x28335e27 },
	{ 0x1cb96c39, 0xf71f7cbc, 0x273ea237, 0x4423b598, 0x04cd3116, 0x974e631f, 0xaed5e5c6, 0x2575073d },
	{ 0xa68def77, 0xf98c8070, 0xfceb3370, 0x0e7a97fb, 0x55a0c759, 0xa56d3b1a, 0xa1564684, 0x20633936 }
},

{
	{ 0xb5789a32, 0xa4c25f4a, 0xba92043a, 0x36b05745, 0x4190514b, 0x001cd2df, 0xb97396de, 0x21946d7f },
	{ 0x8f83fa19, 0x1c37eed2, 0x12095131, 0x7d119f57, 0xbc81ff52, 0x5486466b, 0x00cd81c5, 0x1449dcfa },
	{ 0x890eb34e, 0x980d26d0, 0x888ec324, 0x729c0024, 0xb9f1b26a, 0xca156237, 0xb780f5d4, 0x14cd9927 }
},

{
	{ 0x87e8f641, 0xfebc9619, 0x6afde7cc, 0x1d551869, 0xa80ac94b, 0x62027fa6, 0x027e1348, 0x203c9278 },
	{ 0xe481a49d, 0x66c82871, 0x9d17d77e, 0xcbe5b8bb, 0xa9427444, 0xd32be6ea, 0x0e4c2315, 0x2aee6160 },
	{ 0x2adde582, 0xf377e4d4, 0x9bea23c7, 0x7928bf1e, 0x08624ffb, 0xc73a2bd9, 0xde61a71a, 0x100e6109 }
},

{
	{ 0xc9bd0f8d, 0x0a1c7c96, 0x142ddd3d, 0x4aba642d, 0x4aa9d3bd, 0x8114d400, 0x1f4f33b4, 0x1301adef },
	{ 0x1c833eee, 0xd13eba8c, 0x431dcabf, 0x1a794cd5, 0xc7eec790, 0xdc591423, 0xd34e239f, 0x0e53f1cc },
	{ 0x1a63183b, 0xf9eb01ca, 0x1df04ce5, 0xdacaf893, 0xa0e454f1, 0x53a1a353, 0x91ddb43f, 0x16fec60c }
},

{
	{ 0xb2bd0013, 0x382308f5, 0x20d81fc5, 0x780f733c, 0xa741f880, 0x67268716, 0xf1b2d893, 0x06d6bcb5 },
	{ 0xfa658b69, 0x28c4a27a, 0x4ffb081c, 0xade25c0f, 0x2028281e, 0xc7dbd4dd, 0x239e63f7, 0x070f2dc7 },
	{ 0x96fc6a15, 0x2a6f57e3, 0x73a47c85, 0x9f5bd222, 0x0e806c35, 0x5ef6aa67, 0x82163703, 0x0d582b0b }
},

{
	{ 0x8c966662, 0x9075aaa9, 0xeb5d088a, 0x51e3f2cf, 0x927f4327, 0x82404934, 0x8d8e8020, 0x00ed6cad },
	{ 0xc511732d, 0xa06e6f83, 0x80de5538, 0x952b3fed, 0xec42ed06, 0xc48fb186, 0x32e12f64, 0x2efd9a2a },
	{ 0x8c60162d, 0xecc3078b, 0x504d774f, 0x20f7fc5e, 0x65bdc791, 0x75bfbfeb, 0x2696bc9b, 0x05e97d11 }
},

{
	{ 0xf0d906ed, 0xf26e6d73, 0x3128035e, 0xa1eb4aef, 0xd5e97e0c, 0x2a4e615b, 0xb01ea419, 0x159bfad6 },
	{ 0xd692b7cc, 0x09bacc73, 0x6ad230d3, 0xdf1f8a8b, 0x9a9a4364, 0x97e8e27c, 0x690fe7ad, 0x097f4ca9 },
	{ 0x5d97ee51, 0xcd7e4ec4, 0xf7a2b6b8, 0x31177f1c, 0x307e60a8, 0x2553bb76, 0x59b86530, 0x0afc1771 }
},

{
	{ 0xe2ba1e45, 0xde69c01c, 0x06ce1c71, 0x7e9a38a8, 0x72e0c4ed, 0xd0c0078a, 0x9773641a, 0x06bd9090 },
	{ 0x6470e7d3, 0x7738706d, 0xbf70e510, 0x3e112e72, 0xebdf801e, 0x3d239c90, 0x9c0a56ab, 0x11af2dea },
	{ 0x268c11b4, 0x029a9205, 0x9bdbcad0, 0xffdd08ef, 0xe66e034c, 0x12006c39, 0xbb36f285, 0x055ad5e7 }
},

{
	{ 0x554bd887, 0xd1e76651, 0x9e9daef0, 0xdbb17cdf, 0xf6aced40, 0x0086d6b9, 0x2cdefe3d, 0x15b214b3 },
	{ 0x6a50ef94, 0x14ca0243, 0x1fbd7a9b, 0x9cf19aee, 0x4b508fbd, 0x3e4f0fd5, 0x763ea171, 0x00751f13 },
	{ 0xbe35dd7a, 0xbdab3329, 0x15fce22f, 0x85e230b2, 0xf3d7a62f, 0x9b6e84c2, 0x74e63963, 0x22ef2ef5 }
},

{
	{ 0x2c12d485, 0xb562dd87, 0x777c0683, 0xe62ad8a5, 0x99debc77, 0xf0dcfc36, 0xe5e5b0a4, 0x28662760 },
	{ 0x66e658ba, 0x41f86aa4, 0x527df9e7, 0xa2d9dcdc, 0xb0b23b20, 0x1c0726d8, 0x1ea206f8, 0x1e237b5e },
	{ 0x21cc0e2c, 0xbe4b142e, 0xa1fef0a9, 0x1f8f62b3, 0x82ac28bc, 0xd1285a9c, 0x83d1e5aa, 0x1b45c64f }
},

{
	{ 0x821c35c6, 0x2aa7e53c, 0x66766758, 0x5caf5185, 0x304f7d78, 0x3c7b8d55, 0xe5fcc3b6, 0x2ebff145 },
	{ 0xf83c7de0, 0xf0f348a1, 0x1f0bd348, 0x78d9431d, 0x37597646, 0x33b3dc37, 0x4b7b2e3d, 0x270c9b1d },
	{ 0x1827e1fa, 0xdb25fd2c, 0xd1229d39, 0xe4f9b79e, 0x4ab07287, 0xb42d6ad2, 0x5ec83a8a, 0x2493dd11 }
},

{
	{ 0x7ce4d9c2, 0xf3e6a4da, 0xeb3856f6, 0x94797b07, 0x302b0d80, 0xb25bcdcb, 0x4b6838d5, 0x1fd19c3d },
	{ 0x434add0d, 0xef1c98ed, 0x3ae0f604, 0x93b3d4fd, 0xc9f9aa82, 0xb5ae7f6a, 0xf984d7b1, 0x1a31f46c },
	{ 0xdaeb855e, 0x857855ba, 0x6efa703b, 0x6a0f74a6, 0xf9819c31, 0x27e8d2d0, 0x8f830f63, 0x1cb96ff5 }
},

{
	{ 0x2ca3621f, 0xc01ba7ce, 0xbc1a5733, 0x8248e1e1, 0xfb082e73, 0xfcde5030, 0xf9a9ac28, 0x0d80c632 },
	{ 0x5a209db5, 0x25bbf61d, 0x89edcb86, 0xb95adf7f, 0xc9774f5c, 0xd22ab643, 0xacb61e1c, 0x22b17da9 },
	{ 0x88f0d8e5, 0x11261f28, 0x62adae77, 0x64eb50cd, 0x82e99a8b, 0xaab735b1, 0x5903a4c4, 0x091d7f2e }
},

{
	{ 0x69df7d2a, 0xfb4c4aa5, 0xb15dc3f1, 0x797ca095, 0x66a2f2d3, 0x1b4a814f, 0xed5ea99d, 0x1dbb9509 },
	{ 0xf782bca2, 0x35cbeca4, 0x9b1e733f, 0x5d1708a7, 0x1aabb5bf, 0x8a0e9dfc, 0xb4c806a7, 0x11741968 },
	{ 0x0081d677, 0x66f5ad86, 0xfbb1a293, 0x0d58032f, 0x081b5e61, 0x89548cdd, 0x052b86cc, 0x2302e52d }
},

{
	{ 0x6ce5a48c, 0xd39c8244, 0xa458ff54, 0x0d4ba0b2, 0x2f86e479, 0x049fda6b, 0xcae43f19, 0x17cd1f03 },
	{ 0xa3bbd9dd, 0x2b4425b4, 0x7487da3d, 0x9e165563, 0x7bdd1143, 0x042d72bc, 0xd4fa771c, 0x1e9bfe82 },
	{ 0x69c50958, 0x0b36ce52, 0xcd3edc63, 0x9cf0698a, 0x56e4731e, 0xe8af885b, 0xc4ef2d27, 0x1330da1b }
},

{
	{ 0x0259e031, 0x92fac37c, 0x1a666be6, 0xc56bd5bb, 0x940264e8, 0xf6dd60f7, 0x387aa7d9, 0x100a69f3 },
	{ 0xd3eff3c2, 0xa3a51fcd, 0xa3f2ffbb, 0x3bdb8d10, 0xa20b2942, 0x32dc3f50, 0x81cdf379, 0x19f76909 },
	{ 0x6f7eb2fd, 0xa79c2b6a, 0x9fe7f4a3, 0xa2bf48ee, 0xceda5aa5, 0xe58db2e6, 0x8d6e3c83, 0x1948dc9c }
},

{
	{ 0x25c1e9ba, 0xab8c450e, 0xeaad13a9, 0x5aed2b83, 0x96ae1540, 0x8422fc89, 0x6fe53f14, 0x0fc31c00 },
	{ 0xaccc14be, 0xce4e0bd3, 0x22eb5799, 0xe0a937ef, 0x1e1a648a, 0x406eaa8b, 0x141d971a, 0x017440a0 },
	{ 0x378191ba, 0xcd46dc2b, 0x164cf5ca, 0x8a5abb17, 0x804ba9cd, 0x9818eeb6, 0x4e79d822, 0x26b97701 }
},

{
	{ 0x58f42271, 0x8228fa7c, 0x9891995a, 0xc8e250c4, 0x9cab1350, 0x1f9756f9, 0x907f11d1, 0x05c97566 },
	{ 0xf7db2413, 0x6051d2eb, 0x312d4469, 0x68a7e663, 0xcbe942d0, 0x875e5559, 0x4c6fb4b1, 0x03a1fc05 },
	{ 0x48c20137, 0xdd4cd75a, 0x194118c9, 0xd165a678, 0xd26d54b1, 0x159431d2, 0x90e54a64, 0x061e223a }
},

{
	{ 0xa8d8e7a2, 0x677cff5e, 0x4b0a0d7e, 0x53bbde28, 0x2f97e63d, 0xc9addb42, 0x12e4db2f, 0x0ae9b4fd },
	{ 0x7e2312f3, 0x39b0a9a7, 0x75c83fb8, 0xb52b6588, 0x247c7929, 0x8e9f85cc, 0xe438f90d, 0x02476b2d },
	{ 0x77799ddc, 0xa994e5aa, 0xe2c0701d, 0xf556a7d7, 0x1b0e5373, 0x93ab1ef3, 0x87d8a9a3, 0x196a8a35 }
},

{
	{ 0x87420bea, 0xd1de9b55, 0xf87679ed, 0x9817413d, 0xfe5211a7, 0x70f3de13, 0xb4cd620c, 0x284c2a46 },
	{ 0xd88f01c1, 0x5547180d, 0x023a8ae6, 0xe7e782ec, 0x249d6640, 0x7547dbac, 0xbcfb9a1c, 0x11395652 },
	{ 0x82a58b0f, 0xdefcd154, 0x2a0e9636, 0x8a667d27, 0xaf252e62, 0x94adb950, 0xd34b8f4a, 0x1c6a1eef }
},

{
	{ 0xb66a5a31, 0x58af3cf7, 0xc3d05939, 0x8601ccfc, 0xf61686d9, 0xaa5eab05, 0xe6c15f9d, 0x182178e1 },
	{ 0xfc5d7ae1, 0x228927b5, 0xf23ec7e2, 0x0f6448a8, 0xa7bd5e93, 0x6e4a04a4, 0x79cc7e04, 0x08ec82e5 },
	{ 0x419e8163, 0xf908f84e, 0x9eb4abc6, 0xeccc3905, 0x837dc1a3, 0x8ec0f116, 0x31188bb6, 0x0a7c99eb }
},

{
	{ 0xfbbf2556, 0xd4e01797, 0xf7b43282, 0x5005123a, 0xd4c861a0, 0x14ffe735, 0xa0f78588, 0x253c7977 },
	{ 0xa682e036, 0x738ff92c, 0x31716dc9, 0xad269d52, 0xeea4a6e3, 0x27a66753, 0x6c95ed6f, 0x241b0d22 },
	{ 0xa7d48e38, 0xa94c24ee, 0x5904d866, 0x2c9c20be, 0xb81d549c, 0xcee0f24b, 0xabc68b1d, 0x1800ea0b }
},

{
	{ 0x1383e0ea, 0x6e3de57f, 0x564cb046, 0x38ba7f2c, 0xa7bdca3f, 0x462e7506, 0xd76b2942, 0x1e49f0e6 },
	{ 0xa35ab260, 0xa5ade319, 0xdd4cfc2e, 0xf3c37c88, 0x9e4dba7d, 0x587c75c2, 0x4497f17f, 0x1fb0a368 },
	{ 0xac050538, 0xe775fa5e, 0x1a22b9dd, 0x76acecb1, 0x453f718a, 0x8c484944, 0xdff46562, 0x092d2036 }
},

{
	{ 0xdbc55345, 0x1c709b60, 0xc8aac717, 0x6abd00fe, 0xe592273a, 0xf02a342a, 0x50ec3dc1, 0x0da50afb },
	{ 0xb058fa4b, 0xb3e57790, 0xdb25e48a, 0xdde35da5, 0xe41e0d45, 0xa012062c, 0xe60ab26b, 0x026cf208 },
	{ 0xdf5f120f, 0xc6b65f67, 0x70bdecbd, 0x2f7d3dc7, 0x48d03ac0, 0x8ee8c1fd, 0xa9d0be61, 0x1e59fc8b }
},

{
	{ 0xb550fc77, 0x6cac0295, 0x3675d54b, 0xab06bcba, 0x23c20448, 0x15c76c9f, 0x781938e7, 0x240a0d22 },
	{ 0xe42f0024, 0x922d302c, 0xca48b92c, 0xc93caa18, 0x366a0c2e, 0xd862f6c4, 0xf0ee8633, 0x0b72f6d0 },
	{ 0xbfa47dba, 0x96151661, 0xac991028, 0x6960da29, 0x885c2cbd, 0xcccaa719, 0x9e5f5810, 0x267dde76 }
},

{
	{ 0x78d9f4c2, 0x15b10477, 0xe8eb328c, 0x02c1c5a6, 0xe5894cdd, 0x842004ff, 0xd4d44e2b, 0x06d82bd1 },
	{ 0x1e4401f1, 0x6ea463a1, 0x4d7420bd, 0x0fecdaa7, 0xa71417e0, 0xb4e74ff6, 0x0c35a59c, 0x2770f7f6 },
	{ 0xe6fdb861, 0xe7193728, 0x9229e884, 0x5e6239d8, 0xc9df23b1, 0x68a4e536, 0x2fecec0d, 0x11c260c7 }
},

{
	{ 0xd9a33f44, 0xe404a030, 0xb527609e, 0xf16dcbdf, 0xd736c829, 0x25b1a750, 0x573874b0, 0x0d548516 },
	{ 0xff4e4ff0, 0xff826b73, 0x77710e95, 0x848e6fc6, 0x371b68dc, 0x47601ad8, 0x3ffc8bcf, 0x1dfc76bf },
	{ 0x1bad46c5, 0x8b528d5b, 0x5bc99504, 0xd53c22c6, 0x698a22c5, 0x92c1e5a7, 0x67972aef, 0x2f388bcf }
},

{
	{ 0xd2791341, 0xe70208c6, 0xf46c2f34, 0xb8b31865, 0x81e15d99, 0x37403073, 0x04513ba7, 0x20ad7454 },
	{ 0xb6dc1ef4, 0x001af36d, 0xcf203e14, 0xed814e75, 0x5a53be3c, 0x895b85da, 0xe2fc43d5, 0x0622e756 },
	{ 0x1d68fa55, 0xdf2ed9ea, 0x1c76810c, 0x64aabcc3, 0x9946a3e3, 0xfb21de1f, 0xd1ab9716, 0x1b29e368 }
},

{
	{ 0xef5e25da, 0x7ac219de, 0x36784fc8, 0x6728a4af, 0x84fa376e, 0xd9b8715c, 0x7a6a6cce, 0x0a15c184 },
	{ 0x9496d684, 0x17b09e7f, 0xdb46871f, 0xaa04cf34, 0x58d8b6a0, 0x69c4aff8, 0x78b7f2e9, 0x034489df },
	{ 0xae4745e0, 0x9002c3a5, 0xff05e1bc, 0x1865d0ca, 0xd3a12bcf, 0x4626eb08, 0x4456d142, 0x175c7ac8 }
},

{
	{ 0xcf9ad0a0, 0xae5c263a, 0x96ebe2c7, 0xcd6121b8, 0x81a55461, 0x46f56d51, 0x196e65c6, 0x077c12b7 },
	{ 0xe4454cba, 0xb5c9db81, 0x95d9e675, 0x23e068dc, 0xe3471d7e, 0xa0d1ef7b, 0x0269a922, 0x0ab7865e },
	{ 0x96e9d877, 0x60c44d12, 0x01a9d087, 0x56b38ac7, 0x82af99ca, 0x103cddf8, 0x5ed4a2bb, 0x18f66d27 }
},

{
	{ 0x32f2189f, 0xbac922e8, 0x433b022a, 0x98c7aed1, 0x40153da1, 0x4a836d3a, 0x4f5244d4, 0x2ecf5ec0 },
	{ 0x5d5c8a89, 0xaac5f452, 0xaacfbc20, 0x9d26af1f, 0x01478f73, 0x41b9c237, 0x18a1337e, 0x0d653d25 },
	{ 0x8b8268ec, 0x208171d9, 0x6dc6fe2f, 0x44d67427, 0x4116ce33, 0x72ce99fc, 0x81f91da9, 0x06d70018 }
},

{
	{ 0x5ec9076e, 0xbe9853f0, 0x52ae215d, 0xae88ad61, 0xde59a2e1, 0x77d3ae06, 0x9592f681, 0x17df6d7b },
	{ 0x8fd5ca3b, 0x5b583319, 0x7d71533f, 0x0c57bd0c, 0x9f133cef, 0xd6f50a88, 0xba7b7441, 0x1f5b5088 },
	{ 0x6fad27db, 0xd282f709, 0xfdd0599c, 0x693e93db, 0xc154328f, 0x13bdd0b0, 0x3f3dcf13, 0x021ec3cd }
},

{
	{ 0x9f2f47ff, 0xdfc72a10, 0x57722024, 0xa438eac8, 0xf2edb6ed, 0xce6cb2e9, 0xe3760e58, 0x14690aa8 },
	{ 0xbad1bc14, 0xf30df019, 0x410ce4b3, 0x7ed9f5f2, 0x63e13fbd, 0x0c864029, 0x5b7216e9, 0x25b4b8ae },
	{ 0x8880d8cd, 0x99d01764, 0xce5289e7, 0xdd71cdf2, 0x80034ed2, 0x5b23efdf, 0xeeb72ba8, 0x0f0501d5 }
},

{
	{ 0xb4506200, 0xd634a3b5, 0xbd284f7a, 0x278675c1, 0x5e3efbf3, 0xae1969e3, 0x42e13ad5, 0x139ee8a1 },
	{ 0xdacb1b65, 0xcece0e51, 0x1f9f3487, 0xe9b52a2f, 0x50b1d852, 0xc7324e24, 0xd388d6e1, 0x297c3099 },
	{ 0x6ba3df61, 0xdbd5d0a2, 0x5039dcf4, 0xd6e4b330, 0x0cfc9d73, 0xb5bed725, 0xe45234bf, 0x154f8f82 }
},

{
	{ 0x1172ea2f, 0xd0a74c5d, 0x6f57c66e, 0x78d52d6e, 0x6c363628, 0x484a7428, 0xa94b1acf, 0x0102a653 },
	{ 0xe33a43f2, 0x63c5f87d, 0xe2db2101, 0xeb255053, 0x945a6029, 0xc14cdcd0, 0xddc1fc97, 0x25e2961c },
	{ 0x9133cb86, 0xc1f898ec, 0x73763c62, 0xc000a172, 0xae169f74, 0xdbc3a5a9, 0x54fd6bff, 0x035ae9ea }
},

{
	{ 0xdd4f28e5, 0xe548e4d5, 0x68bacd17, 0x211df072, 0xa5a08129, 0xdafb7a48, 0x34cb19db, 0x0761481c },
	{ 0xfb832339, 0x73677706, 0xf2c63203, 0xdfc95912, 0xac116b6d, 0x4a962b04, 0xb290f70a, 0x2734366d },
	{ 0xf72e31a9, 0x0ebaa166, 0xb05478f7, 0x87135ffe, 0x0b3b3cae, 0x72944831, 0xa02bc2ca, 0x09a8358a }
},

{
	{ 0x4362ede4, 0x8960dedc, 0xefc67450, 0x61aa7584, 0x18c0f73e, 0x545e0a1e, 0x8be2f07b, 0x079ba648 },
	{ 0xbbbef9be, 0x44d41868, 0x563b63d9, 0x592508a5, 0x3822506e, 0x4715dd94, 0x5f23d0cb, 0x30104796 },
	{ 0xa0729681, 0x6a0aed5e, 0x62b2856a, 0x300db4b4, 0x37806e2a, 0xc4045049, 0x096df015, 0x14fa42fc }
},

{
	{ 0xaeca9ed4, 0xad521208, 0x18bf2635, 0x9a5085a5, 0xa53b2630, 0x90cc02fd, 0x122b92c1, 0x23a59bfb },
	{ 0x768cdfb7, 0x03b03acc, 0xa6a85df7, 0x0e0cb6a0, 0x6abba34e, 0x34ba0054, 0xdf58e6a7, 0x20b7127d },
	{ 0x23dc820b, 0x2bf3c555, 0xf605ade8, 0xb936fbdf, 0x37331901, 0x0042efeb, 0xb6349963, 0x17bda016 }
},

{
	{ 0x8eaebb78, 0x5df5cc78, 0x2f0e27dc, 0x3db42ebd, 0x02fe1104, 0x4bb09914, 0xafb64a9f, 0x14401e93 },
	{ 0x8d4ce1ff, 0x95d7b3c4, 0x465270d5, 0x32416460, 0xc0eb68bd, 0x40fd5c14, 0xb12f5ef0, 0x193551f0 },
	{ 0xb2574ddf, 0xf5543299, 0xba029636, 0x3be373fd, 0xb5512040, 0x50fcba20, 0x4e935467, 0x2becdc18 }
},

{
	{ 0x1b89877a, 0x493c51d1, 0x81c032ed, 0xe6fabb7b, 0x9df9b3a0, 0xefa41c6a, 0xa98382ff, 0x1e7bc8c2 },
	{ 0xb18c2782, 0x71e833f2, 0xa6535104, 0x99d5396b, 0x2ebc9eac, 0xd25d8b5e, 0x153c8dfc, 0x18fbc678 },
	{ 0xd49787b0, 0x7688657c, 0xfeb00096, 0xaf605073, 0x889a79b1, 0xc816394b, 0x2728b410, 0x1edcf5ac }
}};


DEVICE_FUNC CONST_MEMORY uint256_g MDS[m][m] = {
{
	{ 0x32d27ec8, 0x604f42e8, 0xa4fb7a43, 0xde013417, 0x64eabb49, 0xaddbe5d3, 0x5859b3c9, 0x0561f135 },
	{ 0x126af791, 0x5ff7e766, 0xe264f78e, 0xb27a2284, 0xe4d45ff6, 0x4c666c54, 0xa6c13a94, 0x0b0b5bae },
	{ 0xb4758035, 0x472f32d4, 0xaf9a7595, 0xbf27a7b0, 0xde6fba72, 0xd0f59bf1, 0xa48dcfa9, 0x2bbd86c6 }
},

{
	{ 0x3eb20b51, 0x62ed11ed, 0x984859a0, 0xbcd19c5a, 0xd436fd09, 0x7b5f8ac0, 0xdb7faf07, 0x16442f3c },
	{ 0x33636cd5, 0x048467d5, 0x2a4beb92, 0x1343f4a3, 0xda41e4b3, 0xcbc30b87, 0x10d9e784, 0x2605073b },
	{ 0x6bac118c, 0xaee6cefb, 0x6045ea43, 0xefe3d494, 0x289a1263, 0xc8c01973, 0x19c525f1, 0x1c57ddf7 }
},

{
	{ 0x6179791e, 0xf65b7aff, 0x71be581f, 0x6df33373, 0x05e11d9e, 0x7df6872f, 0x76efeb1f, 0x09e54f57 },
	{ 0xcf41b328, 0xb6a6c30d, 0x8ef32ccb, 0x56a04890, 0x996976a3, 0xf344078e, 0xc94b50ee, 0x1196ee2e },
	{ 0xc284489c, 0x314d7c90, 0xd852bae5, 0x9ef2a10b, 0xe117cd60, 0xf94d44bb, 0x94e1f7e4, 0x194ecb57 }
}};