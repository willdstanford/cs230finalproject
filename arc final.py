


###########################################################################################
# ARC Transformer
###########################################################################################



###########################################################################################
# imports

import json
import pickle
import random
import requests
import torch
import torch.nn as nn
import zipfile



###########################################################################################
# constants

SPECIAL_TOKENS = {
	'startgrid':10,
	'endgrid':11,
	'startline':12,
	'endline':13,
	'startinput':14,
	'startoutput':15,
	'endofpair':16,
	'padding':17
}

VOCAB_SIZE = 18

MAX_SEQ_LEN = 2 * (30*(30+2)+2) + 3

ORIG_TASKIDS = {
	'training':['a85d4709','c8cbb738','8e1813be','a699fb00','5c2c9af4','44f52bb0','23581191','94f9d214','f9012d9b','4258a5f9','bdad9b1f','d06dbe63','8403a5d5','67e8384a','8731374e','25ff71a9','ecdecbb3','6e19193c','39e1d7f9','ba97ae07','99b1bc43','77fdfe62','50cb2852','4c5c2cf0','d5d6de2d','b91ae062','d037b0a7','93b581b8','025d127b','d2abd087','017c7c7b','28bf18c6','9f236235','c3e719e8','264363fd','6855a6e4','f8c80d96','7837ac64','a68b268e','5c0a986e','890034e9','6773b310','49d1d64f','6cdd2623','36d67576','ed36ccf7','d8c310e9','1f876c06','b60334d2','6c434453','810b9b61','1c786137','74dd1130','ded97339','aedd82e4','5614dbcf','d22278a0','d406998b','a5313dff','6150a2bd','97a05b5b','dbc1a6ce','3bd67248','e9afcf9a','d364b489','2013d3e2','2dc579da','ec883f72','2281f1f4','68b16354','a78176bb','1a07d186','32597951','be94b721','cf98881b','6455b5f5','54d9e175','363442ee','91714a58','a1570a43','a87f7484','6430c8c4','90f3ed37','a5f85a15','95990924','c909285e','a3325580','b94a9452','d631b094','539a4f51','6d0aefbc','e179c5f4','623ea044','a8c38be5','72322fa7','29623171','d13f3404','6e82a1ae','b548a754','7ddcd7ec','7468f01a','29c11459','2dd70a9a','db3e9e38','8f2ea7aa','6fa7a44f','776ffc46','cbded52d','97999447','846bdb03','8d510a79','f25ffba3','c1d99e64','25d487eb','484b58aa','ddf7fa4f','3906de3d','7447852a','c9f8e694','e3497940','46442a0e','c3f564a4','d0f5fe59','4290ef0e','d6ad076f','db93a21d','67a423a3','1190e5a7','6e02f1e3','b190f7f5','f76d97a5','3af2c5a8','239be575','b8cdaf2b','dc0a314f','dc433765','1e0a9b12','10fcaaa3','56dc2b01','4093f84a','508bd3b6','62c24649','de1cd16c','913fb3ed','662c240a','fafffa47','7e0986d6','941d9a10','6b9890af','ff28f65a','80af3007','b27ca6d3','f8b3ba0a','150deff5','952a094c','1b2d62fb','928ad970','d89b689b','3de23699','e21d9049','ba26e723','6a1e5592','39a8645d','56ff96f3','a2fd1cf0','a9f96cdd','ae4f1146','22233c11','780d0b14','3e980e27','1e32b0e9','7f4411dc','4347f46a','6aa20dc0','253bf280','a8d7556c','1f85a75f','ce4f8723','05f2a901','3c9b0459','90c28cc7','2dee498d','3ac3eb23','5168d44c','f1cefba8','a3df8b1e','e6721834','6d0160f0','6cf79266','cce03e0d','7b6016b9','6d75e8bb','e9614598','08ed6ac7','67385a82','22eb0ac0','25d8a9c8','bda2d7a6','b775ac94','d43fd935','cdecee7f','c9e6f938','9565186b','1b60fb0c','fcc82909','178fcbfb','72ca375d','4be741c5','3bdb4ada','27a28665','0520fde7','f5b8619d','a61f2674','5521c0d9','1caeab9d','3befdf3e','ea32f347','e8dc4411','4c4377d9','91413438','b2862040','469497ad','d10ecb37','1f0c79e5','eb281b96','c8f0f002','e76a88a6','bc1d5164','88a10436','6d58a25d','eb5a1d5d','760b3cac','ce22a75a','a65b410d','5582e5ca','0b148d64','e98196ab','73251a56','0a938d79','e48d4e1a','c0f76784','4522001f','d4469b4b','caa06a1f','7fe24cdd','447fd412','d23f8c26','321b1fc6','4938f0c2','82819916','e8593010','85c4e7cd','445eab21','42a50994','bd4472b8','28e73c20','54d82841','b230c067','67a3c6ac','a48eeaf7','ce9e57f2','855e0971','5ad4f10b','feca6190','234bbc79','7df24a62','a416b8f3','a740d043','ef135b50','681b3aeb','50846271','794b24be','75b8110e','09629e4f','b7249182','1f642eb9','ac0a08a4','8efcae92','9ecd008a','5daaa586','1fad071e','a61ba2ce','ce602527','543a7ed5','3eda0437','d07ae81c','f8ff0b80','83302e8f','b1948b0a','1cf80156','963e52fc','3618c87e','d9fac9be','007bbfb7','b8825c91','6f8cd79b','47c1f68c','d90796e8','dc1df850','af902bf9','2c608aff','834ec97d','8eb1be9a','e50d258f','48d8fb45','3aa6fb7a','0ca9ddb6','05269061','c59eb873','0dfd9992','868de0fa','ff805c23','40853293','63613498','b0c4d837','36fdfd69','5117e062','d4a91cb9','746b3537','3428a4f5','bb43febb','60b61512','ea786f4a','9dfd6313','d511f180','9edfc990','aba27056','ae3edfdc','beb8660c','0d3d703e','b9b7f026','f35d900a','fcb5c309','b527c5c6','6ecd11f4','5bd6f4ac','272f95fa','4612dd53','e5062a87','06df4c85','694f12f3','41e4d17e','c444b776','a79310a0','1bfc4729','e509e548','53b68214','d687bc17','e73095fd','496994bd','f2829549','f15e1fac','aabf363d','8d5021e8','3345333e','8a004b2b','b6afb2da','045e512c','673ef223','88a62173','99fa7670','11852cab','bbc9ae5d','444801d8','995c5fa3','9d9215db','f8a8fe49','3631a71a','9172f3a0','3f7978a0','0962bcdd','2204b7a8','b782dc8a','9af7a82c','31aa019c','0e206a2e','7c008303','00d62c1b','d4f3cd78','8e5a5113','44d8ac46','23b5c85d','29ec7d0e','d9f24cd1','f25fbde4','137eaa0f','57aa92db','9aec4887','2bcee788','e40b9e2f','46f33fce','a64e4611','22168020','228f6490','7b7f7511','8be77c9e','dae9d2b5','2bee17df','98cf29f8','e26a3af2'],
	'evaluation':['f0afb749','94414823','dc2e9a9d','f83cb3f6','baf41dbf','93b4f4b3','ff72ca3e','50f325b5','da515329','60a26a3e','14754a24','4ff4c9da','f9d67f8b','5ffb2104','2037f2c7','00dbd492','9c1e755f','6a11f6da','e760a62e','7bb29440','19bb5feb','6ad5bdfd','891232d6','292dd178','67b4a34d','94be5b80','df8cc377','ce8d95cc','72a961c9','6f473927','18419cfa','45bbe264','7c8af763','f8be4b64','e7dd8335','103eff5b','a57f2f04','52fd389e','7d1f7ee8','95a58926','8dae5dfc','2753e76c','c6e1b8da','516b51b7','351d6448','c48954c1','dc2aa30b','712bf12e','cb227835','cd3c21df','20981f0e','03560426','ca8de6ea','e2092e0c','195ba7dc','fc754716','09c534e7','ac0c5833','27a77e38','7e02026e','a680ac02','ac605cbb','5b6cbef5','17b80ad2','4acc7107','67c52801','ce039d91','506d28a5','5a5a2103','0c9aba6e','55783887','ecaa0ec1','929ab4e9','ae58858e','c658a4bd','477d2879','281123b4','12422b43','47996f11','73c3b0d8','137f0df0','94133066','ed98d772','fea12743','e69241bd','64a7c07e','7d419a02','9772c176','b457fec5','310f3251','c92b942c','140c817e','b7999b51','ac3e2b04','3d31c5b3','2546ccf6','626c0bcc','de493100','90347967','88207623','45737921','fb791726','c3202e5a','642d658d','456873bc','782b5218','9b365c51','b9630600','c7d4e6ad','c35c1b4c','60c09cac','d19f7514','8ba14f53','0c786b71','a04b2602','e6de6e8f','7039b2d7','7d18a6fb','4c177718','c97c0139','1e81d6f9','4364c1c4','72207abc','e4075551','31d5ba1a','896d5239','4e45f183','009d5c81','a406ac07','5af49b42','b942fd60','11e1fe23','b7cb93ac','cfb2ce5a','62b74c02','7953d61e','c663677b','96a8c0cd','a8610ef7','0a1d4ef5','69889d6e','a934301b','97239e3d','4f537728','a096bf4d','575b1a71','13713586','8719f442','40f6cd08','12eac192','770cc55f','bc4146bd','0b17323b','ca8f78db','e9bb6954','639f5a19','85b81ff1','551d5bf1','55059096','5783df64','3a301edc','22a4bbc2','4aab4007','f9a67cb5','f823c43c','642248e4','705a3229','ad7e01d0','73182012','e99362f0','c64f1187','4e469f39','e5c44e8f','ccd554ac','7ee1c6ea','e5790162','29700607','9ddd00f0','3194b014','aa18de87','af24b4cc','e1baa8a4','414297c0','e133d23d','1d398264','e88171ec','0e671a1a','8e2edd66','15696249','e7b06bea','48f8583b','7c9b52a0','3391f8c0','f5c89df1','42918530','c074846d','5207a7b5','bf32578f','8b28cd80','fe9372f3','a59b95c0','93c31fbe','1c56ad9f','bf89d739','e78887d1','bd14c3bf','c87289bb','2a5f8217','f21745ec','59341089','833dafe3','505fff84','79369cc6','af22c60d','aab50785','b4a43f3b','b0722778','85fa5666','fd4b2b02','b1fc8b8e','d56f2372','1a2e2828','358ba94e','b20f7c8b','8ee62060','bbb1b8b6','9b2a60aa','25094a63','d5c634a2','0692e18c','d304284e','0f63c0b9','9def23fe','9b4c17c4','27f8ce4f','05a7bcf2','42a15761','c62e2108','817e6c09','ba9d41b8','ea9794b1','8cb8642d','845d6e51','e345f17b','e95e3d8e','9110e3c5','e9b4f6fc','d2acf2cb','0934a4d8','e9c9d9a1','070dd51e','762cd429','da2b0fe3','5289ad53','e21a174a','79fb03f4','c1990cce','20818e16','bcb3040b','2685904e','3490cc26','58743b76','15113be4','d017b73f','cad67732','12997ef3','fd096ab6','5b692c0f','3f23242b','992798f6','1d0a4b61','aa300dc3','e74e1818','4b6b68e5','b15fca0b','f5aa3634','3b4c2228','aa4ec2a5','2b01abd0','21f83797','1acc24af','15663ba9','f3b10344','6ea4a07e','0bb8deee','54db823b','ef26cbf6','f3cdc58f','423a55dc','2697da3f','08573cc6','0a2355a6','256b0a75','50aad11f','f45f5ca7','e66aafb8','1da012fc','1e97544e','d931c21c','68b67ca3','58e15b12','e7a25a18','b0f4d537','332efdb3','16b78196','9c56f360','4cd1b7b2','0607ce86','5b526a93','136b0064','92e50de0','81c0276b','3979b1a8','d37a1ef5','bb52a14b','9bebae7a','66e6c45b','604001fa','981571dc','0becf7df','9356391f','695367ec','50a16a69','ac2e8ecf','a3f84088','212895b5','ea959feb','62ab2642','319f2597','0d87d2a6','dd2401ed','c8b7cc0f','5d2a5c43','4852f2fa','17cae0c1','696d4842','3ed85e70','692cd3b6','d47aa2ff','e619ca6e','1c02dbbe','37d3e8b2','b7fb29bc','48131b3c','2c737e39','f4081712','67636eac','e1d2900e','2c0b0aff','f0df5ff0','d492a647','d94c3b52','e9ac8c9e','e0fb7511','2072aba6','99306f82','6df30ad6','ed74f2f2','1a6449f1','e872b94a','e41c6fd3','31adaf00','73ccf9c2','903d1b4a','1990f7a8','8597cfd7','3ee1011a','917bccba','9f27f097','8a371977','32e9702f','9caba7c3','e633a9e5','e681b708','184a9768','1c0d0a4b','84f2aca1','00576224','84db8fc4','2f0c5170','d4c90558','33b52de3','be03b35f','b7f8a4d8','8fbca751','cf133acc','aee291af','fafd9572','963f59bc','bf699163','759f3fd3','d282b262','5833af48','34b99a2b','f3e62deb','9a4bb226','e7639916','66f2d22f','d4b1c2b1','e57337a4']
}

TORCH_DEVICE = 'cuda'



###########################################################################################
# Data

# the Data class contains functions for downloading, tokenizing, and batching data

class Data:

	def __init__(self,taskids,number_of_instances,batch_size):
		self.load_tasks(taskids)
		train_idxs = [i for i in range(number_of_instances) if i%100!=0 and (i+1)%100!=0]
		validation_idxs = [i for i in range(number_of_instances) if i%100==0]
		test_idxs = [i for i in range(number_of_instances) if (i+1)%100==0]
		train_batch_chunks = [train_idxs[i:i+batch_size] for i in range(0,len(train_idxs),batch_size)]
		self.train_data = self.get_rearc_dataset(taskids,train_batch_chunks)
		self.validation_data = self.get_rearc_dataset(taskids,[validation_idxs])
		self.test_data = self.get_rearc_dataset(taskids,[test_idxs])

	def get_arc_task(self,task_type,taskid):
		response = requests.get("https://raw.githubusercontent.com/fchollet/ARC-AGI/refs/heads/master/data/"+task_type+"/"+taskid+".json")
		task = json.loads(response.content)
		return task

	def get_rearc_task(self,taskid):
		with open('re_arc/tasks/'+taskid+'.json','r') as f:
			task = json.load(f)
		return task

	def load_tasks(self,taskids):
		self.tasks = {}
		for taskid in taskids:
			rearc_task = self.get_rearc_task(taskid)
			arc_task = self.get_arc_task('training',taskid)
			self.tasks[taskid] = {'rearc':rearc_task,'arc':arc_task}

	def get_rearc_batch(self,taskid,idxs):
		all_pairs = self.tasks[taskid]['rearc']
		pairs = [self.tokenize_pair(all_pairs[i]) for i in idxs]
		max_len = max([i.shape[0] for i in pairs])
		padded_pairs = [nn.functional.pad(pair,(0,max_len-pair.shape[0]),value=SPECIAL_TOKENS['padding']) for pair in pairs]
		return torch.stack(padded_pairs).t()

	def get_rearc_dataset(self,taskids,batch_chunks):
		data = []
		for taskid in taskids:
			for idxs in batch_chunks:
				batch_data = self.get_rearc_batch(taskid,idxs)
				data.append((taskid,batch_data))
		return data

	def tokenize_grid(self,pair,in_out):
		tokens = []
		if in_out=='input':
			tokens.append(SPECIAL_TOKENS['startinput'])
		tokens.append(SPECIAL_TOKENS['startgrid'])
		grid = pair[in_out]
		for r in range(len(grid)):
			tokens.append(SPECIAL_TOKENS['startline'])
			for c in range(len(grid[0])):
				s = grid[r][c]
				tokens.append(s)
			tokens.append(SPECIAL_TOKENS['endline'])
		tokens.append(SPECIAL_TOKENS['endgrid'])
		if in_out=='input':
			tokens.append(SPECIAL_TOKENS['startoutput'])
		if in_out=='output':
			tokens.append(SPECIAL_TOKENS['endofpair'])
		return torch.tensor(tokens)

	def tokenize_pair(self,pair):
		return torch.cat([self.tokenize_grid(pair,'input'),self.tokenize_grid(pair,'output')])



###########################################################################################
# TaskModel

# all trained models are built using the TaskModel class

class TaskModel(nn.Module):

	def __init__(self,model_id):
		super().__init__()
		self.token_embeddings = nn.Embedding(VOCAB_SIZE,EMBEDDING_D).to(TORCH_DEVICE)
		self.position_embeddings = nn.Embedding(MAX_SEQ_LEN,EMBEDDING_D).to(TORCH_DEVICE)
		self.embedding_norm = nn.RMSNorm(EMBEDDING_D,eps=1e-5).to(TORCH_DEVICE)
		self.transformers = [nn.TransformerEncoderLayer(EMBEDDING_D,N_HEADS,dim_feedforward=FEEDFORWARD_D).to(TORCH_DEVICE)]*N_LAYERS
		self.linear = nn.Linear(EMBEDDING_D,VOCAB_SIZE).to(TORCH_DEVICE)
		self.model_id = model_id

	def forward(self,x):
		n_tokens,n_batch = x.shape
		x = self.token_embeddings(x)
		x = x + self.position_embeddings(torch.arange(n_tokens).to(TORCH_DEVICE)).unsqueeze(1).repeat(1,n_batch,1)
		x = self.embedding_norm(x)
		mask = torch.triu(torch.ones((n_tokens,n_tokens)),diagonal=1).bool().to(TORCH_DEVICE)
		for transformer in self.transformers:
			x = transformer(x,mask)
		x = self.linear(x)
		return x

	def make_pred(self,x):
		tokens = x
		while True:
			pred = torch.argmax(self.forward(tokens.unsqueeze(0)).squeeze(0)[-1])
			tokens = torch.cat([tokens,pred.unsqueeze(0)])
			if tokens.shape[0]>=MAX_SEQ_LEN:
				return tokens
			if pred==SPECIAL_TOKENS['endofpair']:
				return tokens

	def beam_search(self,x,num_beams=3):
		tokens = x
		beams = [(tokens,0)]*num_beams
		ext = False
		while not ext:
			ext = True
			new_beams = []
			for tokens,prob in beams:
				if tokens.shape[0]<MAX_SEQ_LEN and tokens[-1]!=SPECIAL_TOKENS['endofpair']:
					ext = False
					new_probs,new_tokens = torch.topk(self.forward(tokens.unsqueeze(0)).squeeze(0)[-1],num_beams)
					for new_prob,new_token in zip(new_probs,new_tokens):
						new_beams.append((torch.cat([tokens,new_token.unsqueeze(0)]),prob+new_prob))
				else:
					new_beams.append((tokens,prob))
			beams = sorted(new_beams,key=lambda x:x[1], reverse=True)[0:3]
		return beams[0][0]

	def save(self):
		torch.save(self.state_dict(),self.model_id+'_weights.pth')



###########################################################################################
# Trainer

# the Trainer class contains functions for training and evaluating models

class Trainer:

	def __init__(self,model,optimizer,loss_function):
		self.model = model
		self.optimizer = optimizer
		self.loss_function = loss_function

	def train(self,train_data,validation_data,epochs,verbose=False):
		if verbose:
			print('begin training',self.model.model_id)
		for e in range(epochs):
			train_loss,val_loss = self.train_epoch(train_data,validation_data)
			if verbose:
				print(e+1,train_loss,val_loss)
		if verbose:
			print('end training',self.model.model_id)
		return train_loss,val_loss

	def train_epoch(self,train_data,validation_data):
		self.model.train()
		for (taskid,batch_data) in train_data:
			self.train_batch(batch_data)
		self.model.eval()
		with torch.no_grad():
			train_loss = self.eval_multi_batch(train_data)
			val_loss = self.eval_multi_batch(validation_data)
		self.model.save()
		return train_loss,val_loss

	def train_batch(self,batch_data):
		if TORCH_DEVICE=='cuda':
			torch.cuda.empty_cache()
		self.optimizer.zero_grad()
		pred = self.model(batch_data[:-1,:].to(TORCH_DEVICE))
		pred = torch.flatten(pred,0,1)
		target = batch_data[1:,:].to(TORCH_DEVICE)
		target = torch.flatten(target,0,1)
		train_loss = self.loss_function(pred,target)
		train_loss.backward()
		self.optimizer.step()
		return train_loss.item()

	def eval_batch(self,batch_data):
		with torch.no_grad():
			pred = self.model(batch_data[:-1,:].to(TORCH_DEVICE))
			pred = torch.flatten(pred,0,1)
			target = batch_data[1:,:].to(TORCH_DEVICE)
			target = torch.flatten(target,0,1)
			train_loss = self.loss_function(pred,target)
			return train_loss.item()

	def eval_multi_batch(self,multi_batch):
		with torch.no_grad():
			loss_history = []
			n = 0
			for (taskid,batch_data) in multi_batch:
				loss = self.eval_batch(batch_data)
				loss_history.append(loss*batch_data.shape[1])
				n += batch_data.shape[1]
			return sum(loss_history)/n

	def accuracy(self,batch_data):
		with torch.no_grad():
			correct = 0
			incorrect = 0
			for pair_n in range(batch_data.shape[1]):
				pair = batch_data[:,pair_n]
				output_start_idx = torch.nonzero(torch.eq(pair,SPECIAL_TOKENS['startoutput']))[0][0].item()
				try:
					pad_start_idx = torch.nonzero(torch.eq(pair,SPECIAL_TOKENS['padding']))[0][0].item()
				except:
					pad_start_idx = MAX_SEQ_LEN
				input_seq = pair[:output_start_idx+1].to(TORCH_DEVICE)
				target = pair[:pad_start_idx].to(TORCH_DEVICE)
				pred = self.model.beam_search(input_seq)
				#print(target)
				#print(pred)
				if pred.shape!=target.shape:
					incorrect += 1
					#print('incorrect :(')
				elif pred==target:
					correct += 1
					#print('correct :)')
				else:
					incorrect += 1
					#print('incorrect :(')
			return correct / (incorrect+correct)



###########################################################################################
# tuning

# this function is used to perform hyperparameter tuning

def tuning(taskids,number_of_instances):
	for i in range(number_of_instances):
		# randomize params
		global LEARNING_RATE,WEIGHT_DECAY,BATCH_SIZE,N_EPOCHS,EMBEDDING_D,FEEDFORWARD_D,N_HEADS,N_LAYERS
		LEARNING_RATE = 10**random.randint(-5,-1)
		WEIGHT_DECAY = 10**random.randint(-3,-1)
		BATCH_SIZE = random.choice([10,50,100])
		N_EPOCHS = random.choice([10,20])
		EMBEDDING_D = random.choice([18,32,64,128])
		FEEDFORWARD_D = EMBEDDING_D*random.choice([1,2,4,8])
		N_HEADS = random.choice([i for i in [1,2,3,4,8] if EMBEDDING_D%i==0])
		N_LAYERS = random.choice([2,3,4,6,12])
		# train
		data = Data(taskids,number_of_instances,BATCH_SIZE)
		generic_task_model = TaskModel('generictaskmodel').to(TORCH_DEVICE)
		optimizer = torch.optim.AdamW(generic_task_model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
		loss_function = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['padding'])
		trainer = Trainer(generic_task_model,optimizer,loss_function)
		train_data = data.train_data
		validation_data = data.validation_data
		train_loss,val_loss = trainer.train(train_data,validation_data,N_EPOCHS)
		# result str
		result_str = (
					str(LEARNING_RATE)+'\t'+
					str(WEIGHT_DECAY)+'\t'+
					str(BATCH_SIZE)+'\t'+
					str(EMBEDDING_D)+'\t'+
					str(FEEDFORWARD_D)+'\t'+
					str(N_HEADS)+'\t'+
					str(N_LAYERS)+'\t'+
					str(N_EPOCHS)+'\t'+
					str(train_loss)+'\t'+
					str(val_loss)+'\n')
		with open('tune_results.txt','a') as f:
			f.write(result_str)
		print(result_str)



###########################################################################################
# tuned parameters

# these are the optimal paramaters as determined by hyperparameter tuning

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 10
N_EPOCHS = 10

EMBEDDING_D = 64
FEEDFORWARD_D = EMBEDDING_D*1
N_HEADS = 8
N_LAYERS = 4



###########################################################################################
# train_generic_task_model

# this funciton trains the generic task model using data from all tasks

def train_generic_task_model(taskids,number_of_instances,batch_size):
	data = Data(taskids,number_of_instances,batch_size)
	generic_task_model = TaskModel('generictaskmodel').to(TORCH_DEVICE)
	optimizer = torch.optim.AdamW(generic_task_model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
	loss_function = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['padding'])
	trainer = Trainer(generic_task_model,optimizer,loss_function)
	train_data = data.train_data
	validation_data = data.validation_data
	train_loss,val_loss = trainer.train(train_data,validation_data,N_EPOCHS,True)
	return train_loss,val_loss



###########################################################################################
# train_task_model

# this function trains a model for a specific task, optionally initialized to weights from the generic task model

def train_task_model(taskid,number_of_instances,batch_size,load_generic=False,include_accuracy=False):
	data = Data([taskid],number_of_instances,batch_size)
	if load_generic:
		task_model = TaskModel(taskid+'_taskmodel_gen').to(TORCH_DEVICE)
		task_model.load_state_dict(torch.load('generictaskmodel_weights.pth'))
	else:
		task_model = TaskModel(taskid+'_taskmodel_solo').to(TORCH_DEVICE)
	optimizer = torch.optim.AdamW(task_model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
	loss_function = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['padding'])
	trainer = Trainer(task_model,optimizer,loss_function)
	train_data = [(a,b) for (a,b) in data.train_data if a==taskid]
	validation_data = [(a,b) for (a,b) in data.validation_data if a==taskid]
	test_data = [(a,b) for (a,b) in data.test_data if a==taskid]
	train_loss,val_loss = trainer.train(train_data,validation_data,N_EPOCHS,True)
	if include_accuracy:
		test_accuracy = trainer.accuracy(test_data[0][1])
	else:
		test_accuracy = None
	return train_loss,val_loss,test_accuracy



###########################################################################################
# results

# this code replicates results

def results():
	data = Data(ORIG_TASKIDS['training'],10000,BATCH_SIZE)
	results = {}
	train_loss,val_loss = train_generic_task_model()
	results['base'] = {'train_loss':train_loss,'val__loss':val_loss}
	with open('results.pkl','wb') as f:
		pickle.dump(results,f)
	for taskid in taskids:
		results[taskid] = {}
		train_loss,val_loss,test_accuracy = train_task_model(taskid,True)
		results[taskid]['gen'] = {'train_loss':train_loss,'val__loss':val_loss,'test_accuracy':test_accuracy}
		with open('results.pkl','wb') as f:
			pickle.dump(results,f)
		train_loss,val_loss,test_accuracy = train_task_model(taskid,False)
		results[taskid]['solo'] = {'train_loss':train_loss,'val__loss':val_loss,'test_accuracy':test_accuracy}
		with open('results.pkl','wb') as f:
			pickle.dump(results,f)



###########################################################################################
# example

# this code trains an Independent Task-Specific Model for task 0a938d79 using 1000 instances of the task and a batch size of 10

train_task_model('0a938d79',1000,10)
