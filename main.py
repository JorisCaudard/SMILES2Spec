from smiles2spec.models import Smile2Spec

args = {
    'model_name':"DeepChem/ChemBERTa-10M-MTR",
    'output_activation':'exp',
    'norm_range':None,
    'dropout':0.2,
    'ffn_num_layers':3,
    'ffn_input_dim':199,
    'ffn_hidden_size':2200,
    'ffn_output_dim':1801,
    'ffn_num_layers':3
}

model = Smile2Spec(args)

print(model)

from smiles2spec.train import SIDLoss
import torch

sid = SIDLoss()

target = torch.exp(torch.randn(1, 199))
pred = torch.randn(1, 199)

target = torch.exp(target)
pred = torch.exp(pred)

print(sid(target, pred))

test_smile = 'COC(=O)c1ccc(NC(=O)Cn2c(-c3nnc(CC(C)C)o3)cc3ccccc32)cc1'
HF_token = "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"

from huggingface_hub import login
login(HF_token)

# Load model directly
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")

tokenized_smile = tokenizer(test_smile, return_tensors='pt')

print(tokenized_smile)

model_spectrum = model(**tokenized_smile)

true_spectrum = [0.0003184128493057, 0.0003199769547261, 0.0003202477257843, 0.0003244934754348, 0.0003274225312847, 0.000330637594393, 0.0003320739400228, 0.000337344511725, 0.0003399033685421, 0.000344589543175, 0.000347378751175, 0.0003538723217996, 0.0003615626308345, 0.0003665382445992, 0.0003742768342371, 0.0003757861038086, 0.0003773544796233, 0.0003854879640859, 0.0003903721649951, 0.000392420456537, 0.0003980145671431, 0.0004027628046755, 0.0004058507174153, 0.0004113588988889, 0.0004108004211076, 0.0004176366187338, 0.0005020451383761, 0.0005465193354162, 0.0006375088202736, 0.0006031276616526, 0.0005983544017401, 0.0006214470402014, 0.0006289158856547, 0.0006399862240397, 0.0006518304927955, 0.0006673982035473, 0.0006785976941808, 0.0006802750172319, 0.0006662184914172, 0.0006610721414158, 0.0006644575198006, 0.0006669320852846, 0.0006707614918311, 0.0006641424135032, 0.000652064290935, 0.0006409590666004, 0.000629482040154, 0.0006176291224388, 0.0006133457787833, 0.0006082291214717, 0.0006070784804548, 0.0006050567825145, 0.0005960213113981, 0.0005959246270438, 0.0005957341571377, 0.0005963451936254, 0.0006121187113596, 0.0006141412032824, 0.0006209616865446, 0.000622324552793, 0.000614989301465, 0.0005955014769529, 0.0005892282125077, 0.000590386443585, 0.0005869280633821, 0.0005965603867652, 0.000579652582535, 0.0005830702779772, 0.0005846476268419, 0.0005987853364665, 0.0005985277269099, 0.0005942201515066, 0.0006051361302562, 0.0006058740355008, 0.0006018727112311, 0.0006075656548344, 0.0006279428867532, 0.0006208887462886, 0.0006097761125068, 0.0006076692300022, 0.0006143730556937, 0.0006343063362469, 0.0006374115004785, 0.0006341943395935, 0.0006422379681339, 0.0006517330591197, 0.0006623984097654, 0.0006738708144885, 0.0006860712140063, 0.0007042470504661, 0.0007162002318052, 0.0007205452999038, 0.0007214793804208, 0.0007079646051351, 0.0006835453859499, 0.0006480632719859, 0.0006260163932637, 0.0006051236059362, 0.0005782131629353, 0.0005666412451513, 0.0005522875052361, 0.0005416397664317, 0.0005314899409802, 0.0005121623677523, 0.0004939999971075, 0.0004850713453589, 0.0004801809735398, 0.0004820615711676, 0.0004730454682306, 0.0004699342182233, 0.0004648638315454, 0.0004591700797789, 0.000457198622346, 0.0004540435057303, 0.0004610948472308, 0.0004615266097319, 0.0004784194086516, 0.0004959882050591, 0.0005252423508232, 0.0005437788106196, 0.0005553016342489, 0.0005778196677651, 0.0005820785749559, 0.0005946126613842, 0.0006001138489435, 0.0006107329598098, 0.0006215895995408, 0.0006393846527436, 0.0006507539773361, 0.0006684747701876, 0.0006918030717931, 0.0007127301229005, 0.0007284695116426, 0.0007448980723493, 0.0007800470562937, 0.0008301204049467, 0.0008785346349633, 0.0009376476189224, 0.0009804298952765, 0.0010360683479056, 0.0010823235960402, 0.0011351272863554, 0.0011888248549985, 0.0012218405647823, 0.0012803442599924, 0.0013146704133158, 0.001360113039765, 0.0013914675834392, 0.0014283613987752, 0.001475303178541, 0.0015085275603481, 0.0015382453714177, 0.0015403061153384, 0.0015411136882585, 0.0015362257993843, 0.0015371405528681, 0.001562613073932, 0.001552662641947, 0.0015578986466413, 0.0015696993217218, 0.001565803418643, 0.0015651515199555, 0.0015540890476602, 0.0015597559542298, 0.0015689584242644, 0.001482625020071, 0.0014413529101761, 0.0013742930554487, 0.0013513990128255, 0.0012863969430491, 0.0012403325277072, 0.0012040496742623, 0.0011855959478613, 0.0011838791485744, 0.0012000884729744, 0.0012070886076708, 0.0012324021428884, 0.0012595241411944, 0.0012916569887596, 0.0013091543225361, 0.0013443667475961, 0.0013616963452351, 0.0013732801348946, 0.001388981453985, 0.0014108450279071, 0.0014182999462755, 0.0014339637880054, 0.0014223110570419, 0.0014157330876812, 0.0014165125181159, 0.0014257725595558, 0.0014173072508357, 0.0014034207805618, 0.0013788125914991, 0.0013440565364896, 0.0013091593103929, 0.0012800614358801, 0.0012472866690746, 0.0012058525390131, 0.00118396788664, 0.0011622285733163, 0.0011280609797563, 0.0011272055474003, 0.0011048571258214, 0.0010625595863684, 0.0010400390816518, 0.0010160296875312, 0.0010039383091683, 0.0009930680555399, 0.0009948309239593, 0.0010110766587699, 0.0010185009231823, 0.0010226159506066, 0.0010367209903063, 0.0010367014645686, 0.0010424389216685, 0.0010394197884871, 0.0010185962109805, 0.00099731550283, 0.000965478453721, 0.0009266176128162, 0.0008897774691499, 0.0008680901141085, 0.000852400537446, 0.0008348676139063, 0.0008090454412723, 0.0008254065136648, 0.0008260848222869, 0.0008317530631462, 0.0008530181296371, 0.0008778820380159, 0.0009153621988973, 0.0009491754857083, 0.0009747460888626, 0.0009892156291424, 0.0010241862666892, 0.001040997639572, 0.0010160861778889, 0.0009601071794453, 0.0009510522606277, 0.0009161503643441, 0.0008774027896608, 0.0008556071151024, 0.0008293206740819, 0.000797484764157, 0.0007776659789886, 0.0007835920087721, 0.0007938092471605, 0.0008062950583664, 0.0008140589950028, 0.0008303123974469, 0.0008332868734663, 0.0008524202876651, 0.0008767980477629, 0.0008969922006322, 0.0009145746261961, 0.0009572856222807, 0.0009733113794106, 0.0009986328385766, 0.0010066389568594, 0.0010278948492091, 0.0010615104296166, 0.0010719626031742, 0.0011005646939183, 0.0011159833044656, 0.0011526426859605, 0.0011919348171464, 0.0011914540142816, 0.0012145427420782, 0.0012117914422768, 0.0011908339406981, 0.0011902940848713, 0.0011956807696836, 0.0012023871218737, 0.0011810321661994, 0.0012045851588084, 0.0012255355248127, 0.0012315786926007, 0.0012475379852571, 0.0012678819741986, 0.001277066679726, 0.0012620823951508, 0.0012667131538204, 0.0012507862162195, 0.0012564104471247, 0.0012544447264488, 0.0012611518740005, 0.001270890265891, 0.0012780699168187, 0.0012845019219763, 0.0013021222380828, 0.0013127712845512, 0.0013367657403811, 0.0013373132085226, 0.0013520016626512, 0.0013557021351048, 0.0013331941445774, 0.0013383243023301, 0.0013422554505436, 0.0013441649012812, 0.0013366871363182, 0.001330520607295, 0.0013362394200288, 0.0013241455827026, 0.0013561415067893, 0.0013747741274745, 0.0013455112540037, 0.0013147095951178, 0.0011802575540035, 0.0011573527311305, 0.0011273943897217, 0.001090043280171, 0.0010700563903057, 0.0010442405438868, 0.0010256184443454, 0.000993001457749, 0.0009832221446573, 0.0009534612660826, 0.0009402232657253, 0.0009296341920332, 0.0009358610336626, 0.0009263581551246, 0.0009160065585471, 0.0009015617128617, 0.0009154317761635, 0.000976454980422, 0.001020755320477, 0.0010513399704977, 0.0011401871120118, 0.0012140895516976, 0.0012381018235346, 0.0013231866667755, 0.0013862761828703, 0.0014371982131741, 0.0014635836377894, 0.0015163726719426, 0.0015755486434819, 0.0015734019295336, 0.0016135424998467, 0.0015845372649648, 0.0016122443050587, 0.0016301231608375, 0.0016558523014752, 0.0016789877597845, 0.0017322617244481, 0.0018228671447922, 0.0018965991993813, 0.0020037644299909, 0.002116788204399, 0.0022179318422742, 0.0023328701490787, 0.0024403406544341, 0.0025859526242937, 0.002719587492806, 0.002866668062537, 0.0029882160240785, 0.0031967473103283, 0.0033384270365697, 0.0034644796926664, 0.0036063813627708, 0.0037212025724833, 0.003777874420007, 0.0037814858906172, 0.0038230761308359, 0.0038585354116465, 0.0038887049520747, 0.0039331711598828, 0.0039683263727376, 0.0040445072475493, 0.0040807724967252, 0.0042041270147027, 0.0042916818243359, 0.0043638328544044, 0.0044552346139265, 0.004556114312193, 0.0046093000675321, 0.0047610268851417, 0.0048222552982126, 0.0048610209895377, 0.0048908114264149, 0.0048780992072756, 0.0048726573172104, 0.0048229982774699, 0.0047944834137238, 0.0048178127870436, 0.0047721082858047, 0.0046933900640428, 0.0046645089017263, 0.0046732337917892, 0.0046317844164596, 0.0045780849597387, 0.0045245496775898, 0.0044208868560626, 0.0043232780154941, 0.004126236378332, 0.0040208426395609, 0.0038986142449752, 0.0037949878127077, 0.0037176054703751, 0.0036682260406421, 0.0036244953996973, 0.0036326912171141, 0.0036442097569039, 0.0036424718072439, 0.0036873223201567, 0.0037535603412273, 0.0037733881386112, 0.0038306172441048, 0.0038797517335341, 0.00385805235641, 0.0039350961650636, 0.0039360478214543, 0.0039955540139765, 0.0040455502887753, 0.0041444216610152, 0.0041511759193763, 0.0041569140078159, 0.004194316743792, 0.0041569710416091, 0.0041304339313353, 0.0041549777138519, 0.0041558351742422, 0.0040908915799511, 0.0040278984590786, 0.0039709468063637, 0.0038847055930983, 0.0038029642913455, 0.0037556466719604, 0.0037122578871008, 0.0037148012927414, 0.0036912270439584, 0.0036841684287523, 0.0036703036706773, 0.0036724058521456, 0.0036916280983631, 0.0037920334761908, 0.0037716872331405, 0.0039214128497619, 0.0039424663121117, 0.0040188949923779, 0.0040103306280736, 0.0040084832471499, 0.0040697605475151, 0.0040653373398273, 0.0040229548357101, 0.0039164042062679, 0.0038112602801088, 0.0037278533647321, 0.0035609368539182, 0.003425695637122, 0.003318955740156, 0.0032718738172354, 0.0031164151450718, 0.0030251139286305, 0.0028602001205736, 0.0027429980487154, 0.0026681931485703, 0.0026380587697435, 0.0025867777571754, 0.0025728819968966, 0.0025571870674855, 0.0025845180104682, 0.0026157582640023, 0.0026513627020582, 0.0026735330309854, 0.0026945894160237, 0.0027336462614467, 0.0027286940171244, 0.002708965611588, 0.0027047650189191, 0.0026870843981747, 0.0026543301603257, 0.0025916999466566, 0.0025331717325234, 0.0025221385727837, 0.0024692351022184, 0.0024512743575128, 0.0024571256850551, 0.0024720951203494, 0.0024538504447821, 0.0024725734865189, 0.0024719804314758, 0.0024804313561069, 0.0025186003776484, 0.002540183020199, 0.0026087249599796, 0.0026685783574551, 0.0027089908619023, 0.0027538095390265, 0.0027960329139775, 0.002823351607063, 0.0028390036536247, 0.0029014216215222, 0.0029505237449391, 0.0029895934325072, 0.0030262002682105, 0.0030417544782111, 0.0030556502127591, 0.0030431464964798, 0.0030210517629221, 0.003053429757673, 0.0030095269560075, 0.0030041427828621, 0.0029910243946621, 0.0029877724560665, 0.0029689380197148, 0.0029337410378333, 0.0029209120302665, 0.0028954399071331, 0.0028868953047986, 0.0029328895804579, 0.002936909545352, 0.0029668771736633, 0.0029723950171914, 0.0030064544421275, 0.0030580432098313, 0.0030437780979598, 0.0029958079613183, 0.0029780447909563, 0.0030032809083541, 0.0030101421696085, 0.0030293408214915, 0.0030722697313805, 0.0030946580354852, 0.0031590813675586, 0.0032826346087849, 0.003334781673026, 0.003406079434086, 0.0034501154261543, 0.0035248279439164, 0.0035605960515203, 0.0035999963734515, 0.003602147588525, 0.0035805627002865, 0.0035540246205638, 0.0035576655710392, 0.0035417523907506, 0.0034909087606318, 0.0034926790594856, 0.0035058865648818, 0.0034986971937256, 0.0034413649226097, 0.0033737681438747, 0.0032496299328595, 0.0030905621102611, 0.0028819378535028, 0.002685928688819, 0.0024754038869854, 0.002300833324154, 0.0020945124019438, 0.0019262706366593, 0.0017876077838814, 0.0016660056882769, 0.0015326079046045, 0.0014312059532426, 0.001342812532052, 0.0012541343970705, 0.0011851525963476, 0.0011039672345621, 0.0010530935956144, 0.0010078855629202, 0.0009829750126347, 0.0009671773966659, 0.0009728862130857, 0.001002660199084, 0.0010305382542419, 0.0010921236768453, 0.0011590758722553, 0.0012399711775348, 0.0013358762962417, 0.0014339234757445, 0.0015570004071535, 0.0016524431296376, 0.0017861121377746, 0.0019153951250079, 0.0020539215307724, 0.0022512071078735, 0.0024329627025141, 0.0026407205946198, 0.0028054129171214, 0.0030411458654368, 0.0032334885703853, 0.0034106426799546, 0.0035238603105529, 0.0036002242601014, 0.0036917945689739, 0.0037369823971073, 0.0037812849718872, 0.0038556730312337, 0.0039496546676188, 0.0040026796330304, 0.0040356980666186, 0.0040159939502562, 0.0040058163276189, 0.0040004460349811, 0.0039116599720165, 0.0037731681854621, 0.0035542337759192, 0.00330805944799, 0.0030339878216552, 0.002780469594199, 0.0025289356237722, 0.0023154675278699, 0.0020985885594314, 0.0018981984298149, 0.001715366228301, 0.0015647151117847, 0.0014086733795787, 0.0012557213828163, 0.0011276229765091, 0.0010022236761293, 0.0008991439388641, 0.0007987393271797, 0.0007157422922424, 0.0006280060474206, 0.000577485759843, 0.0005272375083094, 0.0004886633801373, 0.0004436498885239, 0.0003980151341317, 0.0003660022731761, 0.0003404090117091, 0.0003217213966325, 0.0002933249411799, 0.0002728821391236, 0.0002516468598288, 0.000230007133705, 0.0002149311477277, 0.0001987052171411, 0.0001890314098427, 0.0001776720362773, 0.0001678006310583, 0.0001623088571371, 0.0001624586086557, 0.0001614196665744, 0.0001700231457319, 0.0001765134867281, 0.0001921316497429, 0.000205031982247, 0.0002265587808187, 0.0002533451891283, 0.0002869799085252, 0.0003204577619931, 0.0003622784953675, 0.0004088989826662, 0.0004619885742486, 0.000519065623332, 0.0005952916386815, 0.0006604252052641, 0.0007462933213593, 0.0008334547508237, 0.0009273134668305, 0.0010485425783242, 0.0011643814847442, 0.0013065072240784, 0.0014537326607239, 0.0016298275239958, 0.0018052282424445, 0.0019868427048487, 0.0022114191327673, 0.0024561935865121, 0.0027258965744888, 0.0030115090391359, 0.003343626303715, 0.0036731281676079, 0.0040212133800886, 0.0043572216965516, 0.0046432537583289, 0.0048889593113799, 0.0050824472844824, 0.0051972058989464, 0.0052470364249984, 0.0052255497663808, 0.0051443064534928, 0.0050538786098347, 0.0048073723141501, 0.0045178389213895, 0.0042144750403473, 0.0038781981106578, 0.0035427831773396, 0.0032213436304074, 0.0029137856120435, 0.0026262234373175, 0.0023736977426985, 0.0021747616261223, 0.0019755185739891, 0.0017915477586741, 0.0016046965734112, 0.0014589767735118, 0.0013105793377716, 0.0011689955435635, 0.0010402181305792, 0.0009240862410019, 0.0008170136715778, 0.0007315279779039, 0.0006431230336819, 0.0005688847911729, 0.000500851165878, 0.0004438609447979, 0.0003925833568137, 0.0003427353755286, 0.0002969883565692, 0.0002603156829273, 0.0002238581510856, 0.0001974959526623, 0.0001704490328321, 0.0001470725503205, 0.0001298986731768, 0.000111762941366, 9.850241039519306e-05, 8.527600387778286e-05, 7.943325689226477e-05, 6.689976919026636e-05, 6.869153159565074e-05, 3.800944802281428e-05, 6.445001831494717e-05, 4.679502186986963e-05, 4.603884500634747e-05, 4.33276112454261e-05, 3.226555980622743e-05, 3.5827719752747766e-05, 3.3073858533964014e-05, 4.494486623258568e-05, 3.13065372874549e-05, 3.679293788961648e-05, 1.85713479642381e-05, 2.5197270250325603e-05, 1.4858829150576968e-05, 1.9709169778245357e-05, 1.9249368733999492e-05, 7.120626996341372e-06, 1.465804766217332e-05, 8.599010389515883e-06, 1.5815208855688027e-05, 1.6576153230504926e-05, 1.6215858687016382e-05, 1.349768955002817e-05, 1.1319761526699318e-05, 5.29192773635734e-06, 1.9453487812878936e-05, 2.193743863288061e-05, 2.1890883501485054e-05, 2.291234803850322e-05, 3.159704165741566e-05, 3.121877820898947e-05, 5.07119001631061e-05, 5.4297588613422033e-05, 8.140932691801574e-06, 6.051323921838704e-05, 6.235445120305163e-05, 2.5679777512635054e-05, 6.601534719141369e-05, 6.287087380085574e-05, 1.793369883089919e-05, 7.507854579165326e-05, 7.103256432439064e-05, 2.530836433473746e-05, 7.069389353148109e-05, 6.245842466742731e-05, 7.591237811674987e-05, 5.980477994269512e-05, 6.362407329416411e-05, 6.115638050864182e-05, 5.490595276133396e-05, 6.119039698001725e-05, 3.103957653134166e-05, 5.897626133914918e-05, 6.194861115319391e-05, 6.522064700927075e-05, 5.5521633036763e-05, 6.139463035257827e-05, 5.250039884702807e-05, 5.173731401070511e-05, 4.0126629361637006e-05, 4.7332550479569534e-05, 4.4751035465867255e-05, 2.157368584657083e-05, 3.743568350356144e-05, 3.425401929778336e-05, 2.4762086758614694e-05, 3.136811127957353e-05, 3.1128582382615986e-05, 2.7338664098920497e-05, 2.8513896879275484e-05, 2.8256374188486493e-05, 3.1461587862073096e-05, 2.79656504127612e-05, 2.688784609748037e-05, 2.4859761015071395e-05, 2.586185215100199e-05, 2.6451558527123664e-05, 2.4252912760853324e-05, 2.109856873005113e-05, 2.534555305048373e-05, 2.3454815544283716e-05, 2.215285119149004e-05, 2.658331667342637e-05, 1.9753469266140044e-05, 1.887047606986438e-05, 1.6107610646685137e-05, 1.4007981436469435e-05, 1.2057854551120851e-05, 1.1652458788535206e-05, 8.673882431663514e-06, 8.1856379122431e-06, 5.61174965510655e-06, 4.5775356026506416e-06, 3.805262736849886e-06, 3.2938980337621534e-06, 2.334360014171201e-06, 1.9024384006930463e-06, 1.375595133893827e-06, 6.709082548643698e-07, 8.473556690420808e-07, 1.3298372850424042e-06, 1.0668613256761e-06, 1.8041539429445567e-06, 1.5278934347937797e-06, 2.041856661127122e-06, 3.1423422425918428e-06, 5.315798452996226e-06, 6.698437058325142e-06, 9.953169098198335e-06, 1.2061723356950792e-05, 1.609930701936015e-05, 1.4812569326147242e-05, 1.566002623847394e-05, 1.9760969961569702e-05, 2.0022696163731014e-05, 2.559991926230997e-05, 2.803300210731471e-05, 2.6219105171112105e-05, 2.6866813047582832e-05, 3.112911560260816e-05, 3.024368831426697e-05, 3.109128275946436e-05, 2.9873866013443195e-05, 2.85516967264095e-05, 2.7266514540755763e-05, 2.627596200612385e-05, 2.261470254722045e-05, 2.405911804681784e-05, 1.74614587152478e-05, 1.623029574275332e-05, 1.5684052164574503e-05, 1.4678699487268676e-05, 1.3499547014565452e-05, 8.757956337921153e-06, 5.428770741793264e-06, 7.060672326424954e-06, 2.1979260789392085e-06, 2.714985797317069e-06, 1.949664063675087e-06, 7.772883412823345e-07, 3.751176863876316e-07, 9.537548195937351e-07, 1.8698469532841774e-07, 5.8691898227962964e-08, 1.0872622732574705e-06, 1.174571574017562e-06, 9.121719908406114e-09, 4.903767597587412e-07, 7.343118498528666e-07, 1.0322063021511253e-07, 1.033493741599342e-06, 7.094879057677367e-07, 8.664150887663948e-07, 3.552769410458518e-06, 4.596382688479915e-06, 8.608595661095789e-06, 8.624814189353611e-06, 1.2137222297837502e-05, 1.284960051671624e-05, 1.5230394125735572e-05, 1.5877956269269022e-05, 1.4489562436590656e-05, 1.409818656946525e-05, 1.3423627914402404e-05, 1.4963039322196736e-05, 1.7391530344540585e-05, 1.917639964539081e-05, 3.176462680336938e-05, 2.23848878994036e-05, 1.899626131148388e-05, 2.5903568909020783e-05, 2.8578292802309278e-05, 1.4772230477886447e-05, 3.4598212303420744e-05, 3.951084886986271e-05, 3.462418485522648e-05, 3.729450808180843e-05, 3.640093697867355e-05, 3.417256583511604e-05, 2.816999591560513e-05, 2.7813904751962936e-06, 4.4976365758231785e-05, 4.0170963039214215e-05, 3.7845555375784784e-05, 3.596051898556212e-05, 2.443934360069909e-05, 2.13121769182176e-05, 1.963147254292521e-05, 1.8858269410869107e-05, 3.594841417919133e-06, 7.148823163368541e-06, 2.7155867774494053e-06, 2.030898155003076e-06, 1.3219042802204463e-06, 2.055212633313532e-06, 1.483464541449648e-06, 7.84434323155379e-07, 2.414273832705257e-07, 3.807436255980651e-07, 1.5321684303876572e-08, 1.214881994844374e-09, 1.2997770550866384e-08, 2.6223239946371158e-08, 4.092255274086641e-08, 7.456201338971187e-07, 7.510243454914172e-07, 2.9108556709656687e-06, 1.4500657308076134e-05, 1.3646719038498422e-06, 1.205105359268818e-05, 8.477679257399289e-06, 1.5853069841601092e-05, 1.4000927212338314e-05, 1.703573706020468e-05, 1.593705265808859e-05, 5.428364107550003e-06, 6.711971620508733e-06, 1.8618231569407265e-06, 9.742115909885525e-07, 2.794454100297336e-06, 1.2923175160544849e-06, 2.6060543066030585e-07, 3.743689115980472e-07, 2.001783761920537e-08, 5.584188501727826e-09, 3.233698778442776e-08, 2.7311676601527786e-08, 4.2939958253800884e-07, 1.664543167289225e-07, 2.699893178710265e-07, 7.33335376083472e-07, 7.53314680127719e-07, 9.979882885594058e-06, 1.1131941449426906e-06, 1.0238432034331476e-05, 2.7976585319651093e-05, 4.300411081397532e-05, 5.1248508367351234e-05, 4.45259424610688e-05, 4.093071413081891e-05, 3.349129001640144e-05, 3.698799665986709e-05, 3.333844923265368e-05, 2.931545505532972e-05, 1.1768562730586262e-05, 2.6576291127187103e-06, 7.797150440303462e-07, 2.3123570525771e-06, 1.674171681250263e-06, 5.140362668906254e-07, 1.7705020740842753e-06, 1.988129531696983e-07, 2.099156852423353e-05, 1.948989362863513e-05, 5.680873994591111e-05, 3.567339424105845e-05, 4.315471566388454e-05, 4.1160698502687845e-05, 3.649975375847644e-05, 8.09637095825914e-05, 4.769802333714185e-05, 3.660865181818889e-05, 2.737103795561646e-05, 3.258998513851664e-05, 1.4642636724053988e-05, 4.26826161644763e-06, 2.2316014749918583e-05, 1.6228548604733406e-05, 1.3602965609825303e-05, 1.2750806717855948e-05, 2.4008660668252674e-05, 2.1364480015483973e-05, 2.7252807210989985e-06, 1.3527137052130794e-05, 2.6026510674683718e-05, 2.4207028855822764e-05, 3.883517377481721e-06, 3.756248052003929e-05, 3.341103149380787e-05, 4.9474811949827313e-05, 2.2843386006926572e-05, 6.0746263039483936e-05, 8.308966839874101e-05, 6.576229726659365e-05, 6.982593985839063e-05, 8.631015644496215e-05, 0.0001080569776585, 5.075911733807871e-05, 5.018185257547096e-05, 0.0001095824016965, 0.0001085038761363, 9.66050515285882e-05, 8.80748044885312e-05, 9.302814926062776e-05, 0.0001097691028143, 9.62394312747142e-05, 9.973246172863966e-05, 9.844505337893782e-05, 9.356671867382804e-05, 5.628467683341565e-05, 9.331134022829445e-05, 7.732477755161148e-05, 6.043279186673507e-05, 6.940040685803575e-05, 7.407803487907518e-05, 5.875639885120511e-05, 5.7389229000752655e-05, 7.189739464592115e-05, 3.201568409222425e-05, 4.111373102658378e-05, 5.5809875043106205e-05, 4.8929620097502445e-05, 3.0658854550707263e-10, 4.204081501423308e-05, 4.226891629283571e-05, 4.371719594600231e-05, 4.152898892269727e-05, 4.183715541935991e-05, 3.543591841113807e-05, 4.406602830826149e-05, 3.910333213341213e-05, 2.0477032095371016e-05, 3.616280351964755e-05, 4.735810238332326e-05, 4.034633635443093e-05, 4.7629710694179e-05, 4.390665656161816e-05, 3.875638688165899e-05, 3.5175241662975805e-05, 3.557754673655328e-05, 1.919422089320192e-05, 1.0515832287949538e-05, 2.8328450458729783e-05, 3.060135765660011e-05, 3.0026372255729622e-05, 3.217524963449488e-05, 2.734797867079027e-05, 2.7571025846740635e-05, 2.1652938718656103e-05, 1.922745567664399e-05, 2.7997439426689667e-06, 1.7990592929234003e-05, 1.828241821500052e-05, 1.4927771710017354e-05, 1.1994381370876648e-05, 9.789295574880336e-06, 1.1841432669541044e-05, 9.759374444595792e-06, 1.1406243816644543e-06, 3.317900639025109e-06, 6.508620609475335e-06, 1.1870082923110652e-05, 5.487774677303804e-06, 3.0233350125114447e-06, 2.685578867397154e-06, 2.859350268715018e-06, 2.2049757234510914e-06, 8.048416354511065e-10, 1.8233674115272052e-06, 1.0455837173878309e-06, 5.758961219366604e-07, 1.7526069398298954e-06, 1.444560549033639e-06, 1.5618331499022778e-06, 1.823827410126662e-06, 1.5021640820258516e-06, 2.951655816564268e-06, 2.025978575567347e-06, 2.467360507189807e-06, 2.47587585527176e-06, 3.948038550987771e-06, 4.657545136804538e-06, 5.348710153186116e-06, 1.796489411258866e-06, 2.620730178765821e-06, 7.58378781012202e-06, 1.063142534393493e-05, 8.677791238355512e-06, 1.1219037306133062e-05, 1.528049262278687e-05, 1.1587755926669763e-05, 8.363764695888655e-06, 1.3380503443986028e-05, 1.7962431057294532e-05, 1.825702605046776e-05, 2.3038259658597512e-05, 2.812927420601352e-05, 2.9227232085417223e-05, 2.503764166299117e-05, 2.9696005780600016e-07, 2.5119730947990423e-05, 2.9496445490075327e-05, 3.1985455021003785e-05, 2.8570587005501847e-05, 3.0150184074188063e-05, 3.357881535630897e-05, 2.5168770957879365e-05, 3.058945927468004e-05, 4.691982179738469e-05, 4.557469702326109e-05, 4.9246095141497385e-05, 5.287566267377084e-05, 4.855206585184727e-05, 3.557546974985747e-05, 3.4615330183964954e-05, 4.990984821980227e-05, 4.188347502601097e-05, 4.594102636365621e-05, 5.4753027025622336e-05, 4.35423154835496e-05, 3.324174289335597e-05, 4.103253798597548e-05, 3.41863833111167e-05, 4.262253744359462e-05, 4.834212086140301e-05, 4.816573662160315e-05, 4.349118351148056e-05, 8.262328925160043e-06, 8.63423917525425e-05, 4.685350760069404e-05, 4.725303208331245e-05, 4.856437818507581e-05, 4.876642635472967e-05, 4.013158166240063e-05, 3.123969085267947e-05, 4.430650585499264e-05, 4.669641814728311e-05, 4.927998667649802e-05, 3.927023078158669e-05, 3.370369038469663e-05, 3.116826178103877e-05, 4.68066907655264e-05, 3.174720454953162e-05, 2.598814446091579e-05, 2.433458584252553e-05, 3.238607533344277e-05, 1.7104351041375165e-05, 1.5304539004879515e-05, 1.555365924265094e-05, 1.4267199230416018e-05, 1.0848036752664316e-05, 9.050371727416296e-06, 9.352348593548609e-06, 1.0585167835724398e-05, 9.97701585565614e-06, 1.056138998493643e-05, 8.575259294907877e-06, 9.293432394561509e-06, 1.0278629492454875e-05, 1.257411725112376e-05, 1.727387261504566e-05, 1.842652613422836e-05, 2.131876317572221e-05, 2.6332901549488223e-05, 3.057313402924689e-05, 3.4760120243081e-05, 3.745689544282844e-05, 4.510105663369044e-05, 4.6374721613931546e-05, 5.010177060074423e-05, 4.788842940928515e-05, 4.819601190522583e-05, 2.52760324637804e-05, 4.488594110434951e-05, 4.23095376428202e-05, 3.850217699973288e-05, 3.522261387757435e-05, 3.250789768853379e-05, 2.941681971602296e-05, 2.6208129811520583e-05, 2.6543761006888016e-05, 2.561310639924299e-05, 2.486224276249695e-05, 2.3727690344945053e-05, 2.3457580144674832e-05, 2.326994631660842e-05, 2.010229540677522e-05, 1.915353324301276e-05, 1.72935437671012e-05, 1.6693444027749306e-05, 1.3621485252544356e-05, 1.1721802000976457e-05, 9.055406417402e-06, 7.1640761143363e-06, 7.336779881637092e-06, 4.2230311416938963e-07, 3.25905599163094e-06, 3.395436620982292e-06, 2.260786760362469e-06, 2.615923593371243e-06, 1.708582190952035e-07, 1.522919044572277e-07, 1.9514184210479174e-06, 7.241787585908615e-07, 1.066582812619594e-05, 1.3925812270212003e-05, 1.2866286530800346e-05, 7.48835999740588e-06, 2.0322686422652123e-05, 2.4637305303892456e-05, 2.7363580962858345e-05, 3.4363432899196155e-05, 3.763775491060806e-05, 4.0298009446406974e-05, 4.764459416144933e-05, 5.24250151790356e-05, 5.645660727479836e-05, 6.3948016945142e-05, 7.206883298201053e-05, 7.118205210727921e-05, 6.328245112237571e-05, 6.880469084869206e-05, 6.870324320883471e-05, 7.732177510390868e-05, 9.315403370134144e-05, 0.0001005973242571, 0.0001145133041297, 0.000124760112243, 0.0001363711581303, 0.0001540803010313, 0.0001716529438534, 0.000191169175577, 0.0002149816150596, 0.0002399155386322, 0.000270232330844, 0.0002967512168686, 0.0003249321083951, 0.0003543752605877, 0.0003808061557244, 0.0004025932985443, 0.0004174037766467, 0.0004263085862707, 0.0004263811776679, 0.000430399997106, 0.0004189421997076, 0.0004048010585951, 0.0003903040979183, 0.0003664056209552, 0.0003512606170815, 0.0003291997236043, 0.0002936388892347, 0.0002647885293654, 0.0002405359650214, 0.0002239527239951, 0.0001994603370812, 0.0001985701249844, 0.0002081944669007, 0.0002248278217838, 0.0002430287784528, 0.0002638104803071, 0.000287807016849, 0.0003041982416158, 0.0003245583046911, 0.0003415004715319, 0.0003601102364196, 0.0003780220202729, 0.0003892160948227, 0.0004020977077606, 0.0004087848036865, 0.0004132152099123, 0.0004248955436673, 0.0004231913368523, 0.0004269279282085, 0.0004387921991538, 0.0004523222259427, 0.0004861154019861, 0.0005060231909011, 0.0005444901791199, 0.0005880435606864, 0.0006287502203167, 0.0006766159336724, 0.0007215811180878, 0.000757783673246, 0.000805622064937, 0.0008400925871318, 0.0008882973746687, 0.0009393919124392, 0.0009759626124944, 0.0009914651561834, 0.0010060991490126, 0.0010129025558761, 0.0010095806795366, 0.0009848163597317, 0.000955590509539, 0.0009397813944497, 0.0009153684570941, 0.0009373635035243, 0.0009602948913962, 0.0009507055024988, 0.0009422426003346, 0.0009347712646533, 0.0009492704626772, 0.0009900509706934, 0.0010504697549467, 0.001134768453935, 0.0011343010629557, 0.001039732941169, 0.001025620937208, 0.0010393705977969, 0.0010540873410606, 0.0010686734160464, 0.0010862898709622, 0.0010893372490132, 0.00109802016237, 0.0011082517718579, 0.0011265730975239, 0.001157158425229, 0.0011633387119725, 0.0011922756278438, 0.0012128951866945, 0.0012685814031977, 0.0012556298472498, 0.0012706903757613, 0.0012880000053575, 0.0013137526527541, 0.0013777732261385, 0.0013517681047202, 0.0013487500243838, 0.001362192942406, 0.0014083414121474, 0.0014032618843239, 0.0014051405364543, 0.0014259165579454, 0.0014400997410303, 0.0014733956956444, 0.0014807222446773, 0.0014980048415006, 0.0015068227901085, 0.0015258961726396, 0.0017219384760654, 0.0015305578442007, 0.0015027264454487, 0.001479037819322, 0.0014686675435723, 0.0014418196755807, 0.0013601683705907, 0.001309397700545, 0.0012299733434545, 0.0012214243785149, 0.0011038223266037, 0.0010259081393286, 0.0009553641833499, 0.0009072292250303, 0.0008922861189171, 0.0008066781793863, 0.0007603254172075, 0.0007080502011868, 0.0007129037635903, 0.0006546339060896, 0.0006222163786514, 0.0005998625462747, 0.0005672881046593, 0.0005571085903719, 0.0005070592426701, 0.0004809183902003, 0.0004683947938892, 0.0005320940402087, 0.0004355791041765, 0.0004064794862413, 0.0003898053955179, 0.0003770346937594, 0.0003666775754161, 0.0003450941197143, 0.000333024457762, 0.0003200169735558, 0.0003234968373977, 0.0002861003888333, 0.0002821846711363, 0.0002702849567565, 0.000267467134182, 0.0002528048072766, 0.00023792756644, 0.0002246556197166, 0.0002162513595132, 0.0002113154133488, 0.0002035308382431, 0.0001872134143738, 0.0001861488506712, 0.000170748673566, 0.0001628292174934, 0.0001566243861447, 0.0001540889148545, 0.0001490855775996, 0.0001407482683324, 0.0001359990361188, 0.0001321439867159, 0.0001247796097363, 0.0001181292286633, 0.000113019378193, 0.0001075061187201, 0.0001042223288639, 0.0001012452126543, 9.42542309156082e-05, 8.424464756896103e-05, 8.733210209413719e-05, 8.319044023241364e-05, 8.001995985896322e-05, 7.924857272774179e-05, 7.629258218179035e-05, 6.973038562383508e-05, 6.752421327775528e-05, 6.371618114604548e-05, 5.715156565299406e-05, 5.371232884252531e-05, 5.295550852749596e-05, 4.773796763200659e-05, 4.532964777745768e-05, 4.083977774300882e-05, 4.14846248998952e-05, 3.610357110944738e-05, 3.358220023133053e-05, 3.2924314803094784e-05, 2.497105263550189e-05, 2.737869491134067e-05, 2.4509036196977312e-05, 2.006532303949504e-05, 2.1456095521534238e-05, 1.7092801264871417e-05, 1.7771177119441597e-05, 1.5234327757506396e-05, 9.883673016532571e-06, 1.1938947718939845e-05, 1.2962044851966397e-05, 1.224685575245726e-05, 1.119840450412757e-05, 5.544800057208163e-06, 1.1008314365395356e-05, 9.995495854241835e-06, 4.24306867955806e-06, 8.812778539807864e-06, 7.169229958419611e-06, 5.120359532334899e-06, 6.525620568048236e-06, 6.805856906063086e-06, 5.9503944989769085e-06, 5.804324719085403e-06, 4.337471988132179e-06, 4.62039372266153e-06, 3.9588593756840986e-06, 3.717153642966873e-06, 3.644959105087843e-06, 3.251716332667934e-06, 3.207205308788725e-06, 3.336172923221037e-06, 3.042624670132697e-06, 2.375010700073602e-06, 2.4563585315711525e-06, 3.506952961179406e-06, 2.6878840824528206e-06, 1.8731014002764964e-06, 3.207091047747321e-06, 3.3939289123543864e-06, 3.191704158456005e-06, 3.587379214342273e-06, 5.805764905935778e-06, 7.67392432380845e-06, 5.608083359021726e-06, 1.0900514301149452e-05, 1.0091244282262166e-05, 1.0946761049754956e-05, 5.962971899271074e-06, 9.99174247548706e-06, 1.0612346853218931e-05, 1.2360135959544198e-05, 6.46281508320013e-06, 1.190575158954725e-05, 1.209212114399652e-05, 1.0750512655178632e-05, 1.1727928611530864e-05, 1.0250862166538522e-05, 1.2673207646692005e-05, 1.2480597095793608e-05, 1.6043171303188867e-05, 1.4069758444232423e-05, 1.0286517426116369e-05, 1.1582507430280718e-05, 9.450513599423608e-06, 7.920298883190875e-06, 7.189325049280082e-06, 5.986712775532031e-06, 4.521526446963874e-06, 3.4996801016659693e-06, 3.592275695909471e-06, 1.1427453174098105e-06, 3.958798337549131e-07, 2.813407759176433e-07, 2.3589785623528258e-07, 1.9470845255691434e-07, 1.6740309310006077e-07, 1.0220287666812217e-06, 1.0554049496970797e-06, 8.946652186142948e-07, 8.054450035727673e-07, 7.418566986385937e-07, 6.516774337785949e-07, 4.401305690817848e-07, 2.1626686846200286e-07, 2.031297416763717e-07, 2.450287111693438e-06, 3.344358575785085e-06, 4.764578051959138e-06, 5.732904970636469e-06, 5.468851566467962e-06, 8.62315430064978e-06, 1.1291723546761242e-05, 1.2564518417752605e-05, 1.209908287184339e-05, 1.3156174773774398e-05, 1.748722375045909e-05, 1.964122551022766e-05, 2.5622678128550356e-05, 2.7308596164092224e-05, 3.211535085071143e-05, 3.339370712481985e-05, 3.5883122209203805e-05, 3.677395712133005e-05, 4.0662031830116726e-05, 4.217308934379179e-05, 4.299891985922901e-05, 4.482525845080404e-05, 4.618873852858746e-05, 4.495719790963802e-05, 4.860413214868517e-05, 4.627764328605903e-05, 4.391249442964914e-05, 4.056165741711202e-05, 4.004705191983391e-05, 3.97730121671819e-05, 3.626702150931578e-05, 2.9637982235525508e-05, 2.467928810470277e-05, 1.961297681310293e-05, 1.5698635086647203e-05, 1.458351302829622e-05, 1.2709972408252612e-05, 9.180522078445478e-06, 9.56079509077028e-06, 1.0168399078111344e-05, 9.858784205768327e-06, 9.886258784388736e-06, 9.212708358048988e-06, 9.926430329546265e-06, 1.199699778118064e-05, 1.3883750244320684e-05, 1.5028551076542718e-05, 1.9509629609405487e-05, 2.7196823225570088e-05, 3.729333028298375e-05, 4.650089560836007e-05, 4.982977120741514e-05, 5.375891437556952e-05, 6.048985910642886e-05, 6.648235198689959e-05, 7.096398538182001e-05, 8.392251223609416e-05, 8.844652147038245e-05, 9.934582613532672e-05, 0.0001042024657139, 0.0001091627352322, 0.0001087790175656, 0.0001156235288469, 0.0001164478040729, 0.0001143962000071, 0.0001080875614305, 9.478177166390188e-05, 8.937789657122694e-05, 8.868959281973753e-05, 9.003166825948214e-05, 9.690791734941862e-05, 0.000103116678309, 0.0001104803197212, 0.0001179588322305, 0.0001269696727315, 0.0001384100960881, 0.0001495808794591, 0.0001610876077156, 0.0001752147713803, 0.0001886309770109, 0.0002058556182046, 0.0002275865893628, 0.0002515493337219, 0.0002821339249343, 0.0003112597475066, 0.0003347762067591, 0.0003656422254131, 0.0003917873930779, 0.0004148665032116, 0.0004361823736813, 0.0004629356808298, 0.0004925220606031, 0.0005220156279412, 0.0005586149526976, 0.0005971304907642, 0.0006162786817175, 0.0006444517325382, 0.00069075738438, 0.000734048201857, 0.0007669068064847, 0.0008076491669266, 0.0008542972566094, 0.0008929552173823, 0.0009240747960551, 0.0009486647994811, 0.0009626078244951, 0.0009681930445952, 0.00097964685416, 0.0009907708762116, 0.0010113576228069, 0.0010399196991829, 0.0010646582503752, 0.0010865156051773, 0.0011038847608418, 0.001117358622156, 0.0011163976370991, 0.0011145055075535, 0.0011103813588803, 0.0011091960322918, 0.0011031435066168, 0.0010897740296941, 0.001074461493426, 0.00105837945084, 0.001047822079199, 0.0010239858895215, 0.0009825373108597, 0.0009540483574475, 0.0009149104993844, 0.0009020870945932, 0.0008895547087373, 0.0008483129877208, 0.0008141959687253, 0.0007931330959919, 0.0007675625546199, 0.0007313101008417, 0.0007232455270757, 0.0006918853360642, 0.0006632746608353, 0.0006671846104984, 0.0006517505051413, 0.0006618647079702, 0.0006188874755897, 0.0006274775746427, 0.0006274884799419, 0.0006071975505543, 0.0006077199951602, 0.0006023747642811, 0.0006219094716542, 0.0005807362454294, 0.0005837329172807, 0.0005781079655704, 0.0005759573921503, 0.0005829238822892, 0.000562477585867, 0.0005790483240659, 0.0005653682894242, 0.000588126067599, 0.0005687363690725, 0.0005651020220887, 0.0005604199045246, 0.0005679583705181, 0.0006034341124242, 0.0005443150403431, 0.0005403623654193, 0.0005559769713887, 0.0005881302565284, 0.0006056697039671, 0.0005707285596985, 0.0005799445017241, 0.0005933155268748, 0.0006004268741501, 0.0006157983266657, 0.0006043611665344, 0.0005992543463614, 0.0006009578507001, 0.0006091737287346, 0.0006003076701344, 0.0006047721395327, 0.0006095166794518, 0.0006142127014135, 0.0006150668465803, 0.0006079135204701, 0.0006164555776708, 0.0006262616188746, 0.0006369288803836, 0.0006352008819432, 0.0006373779680303, 0.0006336121993302, 0.0006317091788153, 0.000635082024917, 0.0006346584970919, 0.0006484878481299, 0.0006453133382122, 0.0006499522447322, 0.0006567451237447, 0.0006579053086101, 0.0006645181438903, 0.0006554285792788, 0.0006615337677512, 0.0006727620943467, 0.0006791365609506, 0.0006772950740424, 0.0006745942966237, 0.0006757154819096, 0.0006872607273976, 0.000693659876001, 0.0006751027186225, 0.0007090415100856, 0.0007108397992996, 0.0007106851659758, 0.0007145975405744, 0.0007106514727418, 0.0007193657918082, 0.0007278976091261, 0.0007332466178958, 0.0007478682298943, 0.0007542858677725, 0.0007389084164622, 0.0007452769528057, 0.0007528598618123, 0.0007374394520966, 0.0007536075101836, 0.0007355623322693, 0.0007555778522089, 0.0007283465484798, 0.0007329578512572, 0.0007042781720631, 0.0007202523491696, 0.0007444511228466, 0.0007238310122826, 0.0007433069440821, 0.0007778386030511, 0.0007200520133829, 0.0007238630918437, 0.0007395589352867, 0.0007193503746883, 0.0007043821138844, 0.0007513588295442, 0.0007234192694282, 0.0007752735018176, 0.0007392139500458, 0.0007488447817938, 0.0007265053942663, 0.0007088716718928, 0.0006936935548629, 0.0007012648376227, 0.0007175703464665, 0.0007057351916835, 0.0007194630156253, 0.0007327864547461, 0.0007174334310227, 0.0006903889134182, 0.0007140364835727, 0.0007155598189378, 0.0007177440141304, 0.0007193462930321, 0.000702921590396, 0.0006945960036693, 0.0007174776984453, 0.0006964640384871, 0.0006948865913879, 0.000701765723624, 0.000718563739222, 0.0006853047752316, 0.0006256816730155, 0.0006065368387713, 0.0006128019352206, 0.0006381650319309, 0.0006552238783323, 0.0006404781377187, 0.0006268210378068, 0.0006271867574661, 0.0006150478646925, 0.0006180278150394, 0.0006169034266829, 0.0006254022029005, 0.0006225459395083, 0.0006108482728833, 0.00060612026564, 0.0006080856926119, 0.0006140410765915, 0.0006123701706112, 0.0006207584317789, 0.0006147111978414, 0.0005970905596079, 0.0005976195358494, 0.0006001813690489, 0.0006070783404084, 0.0005887213872839]
true_spectrum = torch.tensor(true_spectrum)

print(true_spectrum.shape)
print(sid(model_spectrum, true_spectrum))