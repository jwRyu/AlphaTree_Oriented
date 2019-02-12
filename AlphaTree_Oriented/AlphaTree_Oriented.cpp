#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <assert.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>

using namespace std;

#define OUTPUT_FNAME "C:/Users/jwryu/RUG/2018/AlphaTree/SalienceTree_oriented_1rep.dat"

#define INPUTIMAGE_DIR	"C:/Users/jwryu/Google Drive/RUG/2018/AlphaTree/imgdata/Oriented"
#define INPUTIMAGE_DIR_COLOUR	"C:/Users/jwryu/Google Drive/RUG/2018/AlphaTree/imgdata/Colour" //colour images are used after rgb2grey conversion
#define REPEAT 1

#define max(a,b) (a)>(b)?(a):(b)
#define min(a,b) (a)>(b)?(b):(a)
#define false 0
#define true  1

#define CONNECTIVITY  8
#define NUM_GREYLEVELS	256
#define NULL_CCQUEUE 255

#define A		1.3543
#define SIGMA	-4.2710 
#define B		0.9472
#define M		0.02

#define M_PI 3.1415926535897932384626433

//Memory allocation reallocation schemes
#define TSE 0
#define MAXIMUM 1
int mem_scheme = -1;
double size_init[2] = { -1, 2 };
double size_mul[2] = { 1, 1 };
double size_add[2] = { .1, 0 };

typedef unsigned char uint8;
double RGBweight[3] = { 0.50, 0.5, 0.5 };

double MainEdgeWeight = 1.0;
double OrthogonalEdgeWeight = 1.0;

int lambda;
double omegafactor = 200000;

typedef uint8 pixel;
typedef unsigned long uint32;

//tmp
uint32 numpush = 0;
uint32 numpop = 0;

pixel *gval = NULL, *out = NULL;

double nrmsd;

size_t memuse, max_memuse;

inline void* Malloc(size_t size)
{
	void* pNew = malloc(size + sizeof(size_t));

	memuse += size;
	max_memuse = max(memuse, max_memuse);

	*((size_t*)pNew) = size;
	return (void*)((size_t*)pNew + 1);
}

inline void* Realloc(void* ptr, size_t size)
{
	void* pOld = (void*)((size_t*)ptr - 1);
	size_t oldsize = *((size_t*)pOld);
	void* pNew = realloc(pOld, size + sizeof(size_t));

	if (pOld != pNew)
		max_memuse = max(memuse + size, max_memuse);
	else
		max_memuse = max(memuse + size - oldsize, max_memuse);
	memuse += size - oldsize;

	*((size_t*)pNew) = size;
	return (void*)((size_t*)pNew + 1);
}

inline void Free(void* ptr)
{
	size_t size = *((size_t*)ptr - 1);
	memuse -= size;
	free((void*)((size_t*)ptr - 1));
}

inline void Swap(uint32 *a, uint32 *b)
{
	uint32 tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
}

typedef struct Edge
{
	uint32 p, q;
	pixel level;
	struct Edge *next;
} Edge;

typedef struct CCEdgeBucket
{
	Edge *levelhead[NUM_GREYLEVELS], *leveltail[NUM_GREYLEVELS];
} CCEdgeBucket;

typedef struct CCEdgeQueue // CCEdgeQueue should always be at fixed place (CCEQ for pix n should be in CCrepo[n])
{
	Edge *queue[CONNECTIVITY / 2];
	CCEdgeBucket *bucket[CONNECTIVITY / 2];
	uint32 curSize[CONNECTIVITY / 2];
	uint32 iNode; //index to corresponding tree node
	uint8 minori;
	pixel minlev;
	pixel minlev_ori[CONNECTIVITY / 2];
	struct CCEdgeQueue *next;
} CCEdgeQueue;

typedef struct EdgeQueue
{
	CCEdgeQueue **bucket;
	uint32 cnt;
	uint32 minlev, num_grey_levels, num_levels;
	//curSize
	CCEdgeQueue *CCrepo;
	uint32 CCrepo_cursize, CCrepo_maxsize;
	Edge *Edgerepo;
	uint32 Edgerepo_cursize, Edgerepo_maxsize;
	CCEdgeBucket *bucketrepo;
	uint32 bucketrepo_cursize, bucketrepo_maxsize;
	double ori_weight;
	uint32 levelmap[NUM_GREYLEVELS], wlevelmap[NUM_GREYLEVELS];
	double *level2alpha;
} EdgeQueue;

typedef struct AlphaNode
{
	uint32 parent;
	uint32 area;
	bool filtered; /* indicates whether or not the filtered value is OK */
	double outval;  /* output value after filtering */
	double alpha;  /* alpha of flat zone */
	double sumPix;
	pixel minPix;
	pixel maxPix;
	CCEdgeQueue *ccqueue;
	double orisum[CONNECTIVITY / 2];
	uint8 main_ori; // 0: hor / 1: ver / 2: diag(135deg) / 3: diag(45) / 4: none or insignificant (lower than thr)
} AlphaNode;

typedef struct AlphaTree {
	uint32 imgsize;
	uint32 maxSize, curSize;
	AlphaNode *node;
} AlphaTree;

inline CCEdgeBucket* NewCCEdgeBucket(EdgeQueue *queue)
{
	return &queue->bucketrepo[queue->bucketrepo_cursize++];
}

void CCEdgeQueueInit(EdgeQueue *queue)
{
	uint32 i;
	CCEdgeQueue* CCqueue = queue->CCrepo;

	for (i = 0; i < queue->CCrepo_maxsize; i++)
	{
		CCqueue[i].next = NULL;

		CCqueue[i].queue[0] = NULL;
		CCqueue[i].bucket[0] = NULL;
		CCqueue[i].curSize[0] = 0;
		CCqueue[i].minlev_ori[0] = (pixel)(queue->num_grey_levels - 1);

		CCqueue[i].queue[1] = NULL;
		CCqueue[i].bucket[1] = NULL;
		CCqueue[i].curSize[1] = 0;
		CCqueue[i].minlev_ori[1] = (pixel)(queue->num_grey_levels - 1);
#if CONNECTIVITY == 8
		CCqueue[i].queue[2] = NULL;
		CCqueue[i].bucket[2] = NULL;
		CCqueue[i].curSize[2] = 0;
		CCqueue[i].minlev_ori[2] = (pixel)(queue->num_grey_levels - 1);

		CCqueue[i].queue[3] = NULL;
		CCqueue[i].bucket[3] = NULL;
		CCqueue[i].curSize[3] = 0;
		CCqueue[i].minlev_ori[3] = (pixel)(queue->num_grey_levels - 1);
#endif		
	}
}

inline Edge* NewEdge(EdgeQueue *queue, uint32 p, uint32 q, pixel level)
{
	Edge *ret = &queue->Edgerepo[queue->Edgerepo_cursize++];
	ret->p = p;
	ret->q = q;
	ret->level = level;
	return ret;
}

inline void PushCCEdgeQueueToBucket(EdgeQueue *queue, CCEdgeQueue *CCqueue, pixel edgeSalience) //put them in any bucket first, take care of ori_weight later. only for Phase1
{
	CCqueue->next = queue->bucket[0]; //Store them at the first bucket(arbitrary chosen), and sort them later (because you don't need a sorted queue at Phase1)
	queue->bucket[0] = CCqueue;
	queue->cnt++;
}

inline void ConvListToBucket(EdgeQueue *queue, CCEdgeQueue *CCqueue, CCEdgeBucket *bucket, uint8 ori)
{
	Edge *p, *q;
	uint32 i;
	p = CCqueue->queue[ori];
	CCqueue->bucket[ori] = bucket;
	for (i = 0; i < CCqueue->minlev_ori[ori]; i++)
		bucket->levelhead[i] = NULL;
	while(1)
	{
		bucket->levelhead[i] = p;
		for (q = p->next; q && (p->level == q->level); q = p->next)
			p = q;
		bucket->leveltail[i] = p;
		p->next = NULL;
		p = q;
		i++;
		if (p)
		{
			while (i != p->level)
				bucket->levelhead[i++] = NULL;
		}
		else
		{
			while (i < queue->num_grey_levels)
				bucket->levelhead[i++] = NULL;
			break;
		}
	}
}

inline void PushEdgeToCCBucket(Edge *e, CCEdgeBucket *bucket)
{
	e->next = bucket->levelhead[e->level];
	if(!e->next)
		bucket->leveltail[e->level] = e;
	bucket->levelhead[e->level] = e;
}

inline void PushListToCCBucket(Edge *list, CCEdgeBucket *bucket)
{
	Edge *p, *q, *r;
	for (p = list; p; p = q)
	{
		r = p;
		for (q = p->next; q && p->level == q->level; q = q->next)
			p = q;
		p->next = bucket->levelhead[p->level];
		if (!p->next)
			bucket->leveltail[p->level] = p;
		bucket->levelhead[p->level] = r;	
	}
}
/*
uint8 testcceqbucket(CCEdgeQueue *cceq)
{
	uint8 test0 = 0, test1 = 0;
	if (cceq->minlev_ori[0] < 255)
	{
		if (cceq->bucket[0])
			test0 = (cceq->bucket[0]->levelhead[cceq->minlev_ori[0]] == NULL) || 
			(cceq->bucket[0]->levelhead[cceq->minlev_ori[0]] == (Edge*)(0xcdcdcdcdcdcdcdcd));
		else
			test0 = cceq->queue[0]->level != cceq->minlev_ori[0];
	}
	if (cceq->minlev_ori[1] < 255)
	{
		if (cceq->bucket[1])
			test1 = (cceq->bucket[1]->levelhead[cceq->minlev_ori[1]] == NULL) ||
			(cceq->bucket[1]->levelhead[cceq->minlev_ori[1]] == (Edge*)(0xcdcdcdcdcdcdcdcd));
		else
			test1 = cceq->queue[1]->level != cceq->minlev_ori[1];
	}

	return  test0 || test1;
}
*/


void PushEdgeToCCEdgeQueue(EdgeQueue *queue, CCEdgeQueue *CCqueue, Edge *e, uint8 ori) // put an Edge 'e' in a CCEdgeQueue 'CCqueue'
{
	Edge *p, *q;

	CCqueue->minlev_ori[ori] = MIN(CCqueue->minlev_ori[ori], e->level);
	if (CCqueue->curSize[ori] < queue->num_grey_levels - 1)
	{
		p = CCqueue->queue[ori];
		if ((p == NULL) || (e->level <= p->level))
		{
			e->next = p;
			CCqueue->queue[ori] = e;
		}
		else
		{
			while (p && e->level > p->level)
			{
				q = p;
				p = p->next;
			}
			e->next = p;
			q->next = e;
		}
	}
	else
	{
		if (CCqueue->curSize[ori] == queue->num_grey_levels - 1) // change linked list to bucket sorted list
			ConvListToBucket(queue, CCqueue, NewCCEdgeBucket(queue), ori);
		PushEdgeToCCBucket(e, CCqueue->bucket[ori]);
	}
	CCqueue->curSize[ori]++;
//	numpush++;//tmp
}

void PushEdgeToIncidentCCs_Newpixel(AlphaTree *tree, EdgeQueue *queue, uint32 p, uint32 q, uint32 s, pixel edgeSalience, uint8 ori) //p is always a root, p & q never have a common ancestor. For Phase1.
{
	Edge *e;

	e = NewEdge(queue, p, q, edgeSalience);
	
	//add e on CCEdgeQueue of node[p], which must be empty
	e->next = NULL;
	tree->node[p].ccqueue->queue[ori] = e;
	tree->node[p].ccqueue->curSize[ori] = 1;
//tmp//	numpush++;
	tree->node[p].ccqueue->minlev_ori[ori] = edgeSalience;
	tree->node[p].ccqueue->iNode = p;
	PushCCEdgeQueueToBucket(queue, tree->node[p].ccqueue, edgeSalience); //add a CCEdgeQueue to the EdgeQueue

	e = NewEdge(queue, p, q, edgeSalience);
	PushEdgeToCCEdgeQueue(queue, tree->node[s].ccqueue, e, ori);
}

void PushEdgeToIncidentCCs(AlphaTree *tree, EdgeQueue *queue, uint32 p, uint32 q, uint32 r, uint32 s, pixel edgeSalience, uint8 ori)
{
	Edge *e;
	e = NewEdge(queue, p, q, edgeSalience);
	PushEdgeToCCEdgeQueue(queue, tree->node[r].ccqueue, e, ori);

	e = NewEdge(queue, p, q, edgeSalience);
	PushEdgeToCCEdgeQueue(queue, tree->node[s].ccqueue, e, ori);
}

inline void CCEdgeQueueMerge_Ori(EdgeQueue *queue, CCEdgeQueue *q1, CCEdgeQueue *q2, uint8 ori)
{
	uint32 n1 = q1->curSize[ori];
	uint32 n2 = q2->curSize[ori];
	uint32 i;
	Edge *p, *q, *r;

	q2->curSize[ori] += q1->curSize[ori];
	q2->minlev_ori[ori] = MIN(q1->minlev_ori[ori], q2->minlev_ori[ori]);
	if (q2->bucket[ori])
	{
		if (q1->bucket[ori]) //Bucket + Bucket
		{			
			for (i = q1->minlev_ori[ori]; i < queue->num_grey_levels; i++)
			{
				if (q1->bucket[ori]->levelhead[i])
				{
					if (q2->bucket[ori]->levelhead[i])
						q1->bucket[ori]->leveltail[i]->next = q2->bucket[ori]->levelhead[i];
					else
						q2->bucket[ori]->leveltail[i] = q1->bucket[ori]->leveltail[i];
					q2->bucket[ori]->levelhead[i] = q1->bucket[ori]->levelhead[i];
				}				
			}
		}
		else //List + Bucket
			PushListToCCBucket(q1->queue[ori], q2->bucket[ori]);

	}
	else
	{
		if (q1->bucket[ori]) //Bucket + List
		{
			PushListToCCBucket(q2->queue[ori], q1->bucket[ori]);
			q2->bucket[ori] = q1->bucket[ori];
		}
		else if (n2 + n1 >= queue->num_grey_levels) //List + List => Bucket
		{
			if (n2)
			{
				ConvListToBucket(queue, q2, NewCCEdgeBucket(queue), ori);
				PushListToCCBucket(q1->queue[ori], q2->bucket[ori]);
			}
			else
			{
				ConvListToBucket(queue, q1, NewCCEdgeBucket(queue), ori);
				q2->bucket[ori] = q1->bucket[ori];
			}
		}
		else // List + List => List
		{
			p = r = q2->queue[ori];
			q = q1->queue[ori];
			if (q && (!p || p->level > q->level))
			{
				r = q;
				q = q->next;
			}
			else if (p)
				p = p->next;

			q2->queue[ori] = r;
			while (p && q)
			{
				if (p->level < q->level)
				{
					r->next = p;
					p = p->next;
				}
				else
				{
					r->next = q;
					q = q->next;
				}
				r = r->next;
			}									
			if (p)
				r->next = p;
			else if(r)
				r->next = q;
		}
	}
}

CCEdgeQueue* CCEdgeQueueMerge(EdgeQueue *queue, CCEdgeQueue *q1, CCEdgeQueue *q2)
{
	CCEdgeQueueMerge_Ori(queue, q1, q2, 0);
	CCEdgeQueueMerge_Ori(queue, q1, q2, 1);
#if CONNECTIVITY == 8
	CCEdgeQueueMerge_Ori(queue, q1, q2, 2);
	CCEdgeQueueMerge_Ori(queue, q1, q2, 3);
#endif
	return q2;
}

void EdgeQueueInit(EdgeQueue *queue, uint32 img_width, uint32 img_height, double ori_weight)
{
	double *level, *wlevel;
	uint32 num_levels;
	uint32 i, j;
	queue->num_grey_levels = NUM_GREYLEVELS;
	queue->ori_weight = ori_weight;

	//preallocate repositories
	queue->CCrepo_cursize = queue->Edgerepo_cursize = queue->bucketrepo_cursize = 0;
	queue->CCrepo_maxsize = img_width * img_height;
	queue->Edgerepo_maxsize = (CONNECTIVITY / 2) * (img_width * (img_height - 1) + (img_width - 1) * img_height);
	queue->bucketrepo_maxsize = (uint32)((queue->Edgerepo_maxsize + NUM_GREYLEVELS - 1) / NUM_GREYLEVELS);
	queue->CCrepo = (CCEdgeQueue*)Malloc(queue->CCrepo_maxsize * sizeof(CCEdgeQueue)); //Estimate the number of 0-CCs, and reduce the repo size later
	queue->Edgerepo = (Edge*)Malloc(queue->Edgerepo_maxsize * sizeof(Edge));
	queue->bucketrepo = (CCEdgeBucket*)Malloc(queue->bucketrepo_maxsize * sizeof(CCEdgeBucket));

	level = (double*)Malloc((NUM_GREYLEVELS + 1) * sizeof(double));
	wlevel = (double*)Malloc((NUM_GREYLEVELS + 1) * sizeof(double));
	for (i = 0; i < NUM_GREYLEVELS; i++)
	{
		level[i] = (double)i;
		wlevel[i] = ((double)i) / ori_weight;
	}
	level[NUM_GREYLEVELS] = (double)NUM_GREYLEVELS;
	wlevel[NUM_GREYLEVELS] = (double)NUM_GREYLEVELS;

	//Count the total number of levels (number of greylevels(diff) + number of weighted levels - redundant levels)
	num_levels = i = j = 0;
	while (i < NUM_GREYLEVELS || j < NUM_GREYLEVELS)
	{
		if (level[i] == wlevel[j])
		{
			queue->levelmap[i++] = num_levels;
			queue->wlevelmap[j++] = num_levels++;
		}
		else if (level[i] > wlevel[j])
			queue->wlevelmap[j++] = num_levels++;
		else
			queue->levelmap[i++] = num_levels++;
	}
	queue->num_levels = num_levels;
	queue->level2alpha = (double*)Malloc(num_levels * sizeof(double));
	for (i = 0; i < NUM_GREYLEVELS; i++)
	{
		queue->level2alpha[queue->levelmap[i]] = (double)i;
		queue->level2alpha[queue->wlevelmap[i]] = (double)i / ori_weight;
	}

	queue->bucket = (CCEdgeQueue**)Malloc(num_levels * sizeof(CCEdgeQueue*));
	for (i = 0; i < num_levels; i++)
		queue->bucket[i] = NULL;

	Free(level);
	Free(wlevel);
}



void EdgeQueueDelete(EdgeQueue *queue)
{
	Free(queue->CCrepo);
	Free(queue->Edgerepo);
	Free(queue->bucketrepo);
	Free(queue->bucket);
	Free(queue->level2alpha);
}

AlphaTree *CreateAlphaTree(int imgsize, int treesize) {
	AlphaTree *tree = (AlphaTree*)Malloc(sizeof(AlphaTree));
	tree->imgsize = imgsize;
	tree->maxSize = treesize;  /* potentially twice the number of nodes as pixels exist*/
	tree->curSize = imgsize;    /* first imgsize taken up by pixels */
	tree->node = (AlphaNode*)Malloc((tree->maxSize) * sizeof(AlphaNode));
	return tree;
}

void DeleteTree(AlphaTree *tree) {
	Free(tree->node);
	Free(tree);
}

void AlphaNodeInit(AlphaTree *tree, const pixel *img, const pixel **ori_imgs, EdgeQueue *queue)
{
	uint32 p;

	for (p = 0; p < tree->imgsize; p++)
	{
		tree->node[p].parent = p;
		tree->node[p].alpha = 0;
		tree->node[p].area = 1;
		tree->node[p].minPix = tree->node[p].maxPix = img[p];
		tree->node[p].sumPix = (double)img[p];
		tree->node[p].ccqueue = &queue->CCrepo[p];
		tree->node[p].orisum[0] = ori_imgs[0][p];
		tree->node[p].orisum[1] = ori_imgs[1][p];
#if CONNECTIVITY == 8
		tree->node[p].orisum[2] = ori_imgs[2][p];
		tree->node[p].orisum[3] = ori_imgs[3][p];
//			for (i = 0; i < CONNECTIVITY / 2; i++)
//				tree->node[p].orisum[i] = ori_imgs[i][p];
#endif
	}
}

uint32 NewAlphaNode(AlphaTree *tree, double alpha) {
	AlphaNode *node = tree->node + tree->curSize;
	if (tree->curSize == tree->maxSize)
	{
		printf("Reallocating...\n");
		tree->maxSize = min(tree->imgsize * 2, tree->maxSize + (int)(tree->imgsize * size_add[mem_scheme]));

		tree->node = (AlphaNode*)Realloc(tree->node, tree->maxSize * sizeof(AlphaNode));
		node = tree->node + tree->curSize;
	}
	
//	node->area = 0;
	node->alpha = alpha;
	node->parent = tree->curSize;
	return tree->curSize++;
}

int FindRoot(uint32 *root, uint32 p) {
	int r = p, i, j;

	while (root[r] != r) {
		r = root[r];
	}
	i = p;
	while (i != r) {
		j = root[i];
		root[i] = r;
		i = j;
	}
	return r;
}
/*
int FindRoot1(AlphaTree *tree, uint32 *root, uint32 p) {
	uint32 r = p, i, j;

	while (root[r] != r) {
		r = root[r];
	}
	i = p;
	while (i != r) {
		j = root[i];
		root[i] = r;
		tree->node[i].parent = r;
		i = j;
	}
	return r;
}
*/
void Union0(AlphaTree *tree, uint32 *root, uint32 p, uint32 q) //both p and q are always roots. p and q never have a common root. p does not have a CCEqueue. use on Phase1 only.
{
	tree->node[p].parent = q;
	root[p] = q;
	tree->node[q].area += tree->node[p].area;
	tree->node[q].sumPix += tree->node[p].sumPix;
	tree->node[q].minPix = MIN(tree->node[p].minPix, tree->node[q].minPix);
	tree->node[q].maxPix = MAX(tree->node[p].maxPix, tree->node[q].maxPix);
	tree->node[q].orisum[0] += tree->node[p].orisum[0];
	tree->node[q].orisum[1] += tree->node[p].orisum[1];
#if CONNECTIVITY == 8
	tree->node[q].orisum[2] += tree->node[p].orisum[2];
	tree->node[q].orisum[3] += tree->node[p].orisum[3];
	//tree->node[p].ccqueue = tree->node[q].ccqueue; // only roots have a valid CCEQ
#endif
}

void Union1(AlphaTree *tree, EdgeQueue *queue, uint32 *root, uint32 p, uint32 q) //both p and q are always roots. p and q never have a common root. No main ori comput. Use on Phase1
{
	tree->node[p].parent = q;
	root[p] = q;
	tree->node[q].area += tree->node[p].area;
	tree->node[q].sumPix += tree->node[p].sumPix;
	tree->node[q].minPix = MIN(tree->node[p].minPix, tree->node[q].minPix);
	tree->node[q].maxPix = MAX(tree->node[p].maxPix, tree->node[q].maxPix);
	tree->node[q].orisum[0] += tree->node[p].orisum[0];
	tree->node[q].orisum[1] += tree->node[p].orisum[1];
#if CONNECTIVITY == 8
	tree->node[q].orisum[2] += tree->node[p].orisum[2];
	tree->node[q].orisum[3] += tree->node[p].orisum[3];
#endif

	CCEdgeQueueMerge(queue, tree->node[p].ccqueue, tree->node[q].ccqueue);
}

void Union2(AlphaTree *tree, EdgeQueue *queue, uint32 *root, uint32 p, uint32 q) //both p and q are always roots. p and q never have a common root. use on Phase2
{
	AlphaNode *pNode, *qNode;
	pNode = &tree->node[p];
	qNode = &tree->node[q];
	pNode->parent = q;
	root[p] = q;
	qNode->area += pNode->area;
	qNode->sumPix += pNode->sumPix;
	qNode->minPix = MIN(pNode->minPix, qNode->minPix);
	qNode->maxPix = MAX(pNode->maxPix, qNode->maxPix);
	qNode->orisum[0] += pNode->orisum[0];
	qNode->orisum[1] += pNode->orisum[1];
#if CONNECTIVITY == 8
	qNode->orisum[2] += pNode->orisum[2];
	qNode->orisum[3] += pNode->orisum[3];
#endif

#if CONNECTIVITY == 4
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[1 - qNode->main_ori])
		qNode->main_ori = 1 - qNode->main_ori;
#elif CONNECTIVITY == 8
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[0])
		qNode->main_ori = 0;
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[1])
		qNode->main_ori = 1;
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[2])
		qNode->main_ori = 2;
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[3])
		qNode->main_ori = 3;
#endif
	if (queue->bucket[queue->minlev] == qNode->ccqueue)
		CCEdgeQueueMerge(queue, pNode->ccqueue, qNode->ccqueue);
	else
	{
		pNode->ccqueue->iNode = q;
		qNode->ccqueue->iNode = p;
		qNode->ccqueue = CCEdgeQueueMerge(queue, qNode->ccqueue, pNode->ccqueue);
	}
	
}

void Union3(AlphaTree *tree, EdgeQueue *queue, uint32 *root, uint32 p, uint32 q, uint32 r) // r <- p + q
{
	AlphaNode *pNode, *qNode, *rNode;
	pNode = &tree->node[p];
	qNode = &tree->node[q];
	rNode = &tree->node[r];
	pNode->parent = r;
	qNode->parent = r;
	root[p] = r;
	root[q] = r;
	rNode->area = pNode->area + qNode->area;
	rNode->sumPix = pNode->sumPix + qNode->sumPix;
	rNode->minPix = MIN(pNode->minPix, qNode->minPix);
	rNode->maxPix = MAX(pNode->maxPix, qNode->maxPix);
	rNode->orisum[0] = pNode->orisum[0] + qNode->orisum[0];
	rNode->orisum[1] = pNode->orisum[1] + qNode->orisum[1];
#if CONNECTIVITY == 8
	rNode->orisum[2] += pNode->orisum[2] + qNode->orisum[2];
	rNode->orisum[3] += pNode->orisum[3] + qNode->orisum[3];
#endif

	rNode->main_ori = qNode->main_ori;
#if CONNECTIVITY == 4
	if (rNode->orisum[rNode->main_ori] < rNode->orisum[1 - rNode->main_ori])
		rNode->main_ori = 1 - rNode->main_ori;
#elif CONNECTIVITY == 8
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[0])
		qNode->main_ori = 0;
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[1])
		qNode->main_ori = 1;
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[2])
		qNode->main_ori = 2;
	if (qNode->orisum[qNode->main_ori] < qNode->orisum[3])
		qNode->main_ori = 3;
#endif

	if (queue->bucket[queue->minlev] == qNode->ccqueue)
		rNode->ccqueue = CCEdgeQueueMerge(queue, pNode->ccqueue, qNode->ccqueue);
	else
		rNode->ccqueue = CCEdgeQueueMerge(queue, qNode->ccqueue, pNode->ccqueue);
	rNode->ccqueue->iNode = r;
}

inline void PushEdge_NewPixel(AlphaTree *tree, EdgeQueue *queue, uint32 p, uint32 q, const pixel *img, uint32 *root, uint8 orientation) // p is a new pixel
{
	uint32 s = FindRoot(root, q); // no need to find root of p
	pixel edgeSalience = abs((int)img[p] - (int)img[q]);
	if (edgeSalience == 0)
		Union0(tree, root, p, s);
	else
		PushEdgeToIncidentCCs_Newpixel(tree, queue, p, q, s, edgeSalience, orientation);
}

inline void PushEdge_OldPixel(AlphaTree *tree, EdgeQueue *queue, uint32 p, uint32 q, const pixel *img, uint32 *root, uint8 orientation) 
{
	uint32 r = FindRoot(root, p);
	uint32 s = FindRoot(root, q);
	pixel edgeSalience;
	if (r != s)
	{
		if (r > s)
			Swap(&r, &s);
		edgeSalience = abs((int)img[p] - (int)img[q]);
		if (edgeSalience == 0)
			Union1(tree, queue, root, r, s);
		else
			PushEdgeToIncidentCCs(tree, queue, p, q, r, s, edgeSalience, orientation);
	}
}

inline uint8 SetMainOri(AlphaTree *tree, CCEdgeQueue *cceq)
{
	uint32 inode = cceq->iNode;
	uint8 mainori = 0;
	double maxsum = tree->node[inode].orisum[0];
	if (tree->node[inode].orisum[1] > maxsum)
	{
		maxsum = tree->node[inode].orisum[1];
		mainori = 1;
	}
#if CONNECTIVITY == 8
	if (tree->node[inode].orisum[2] > maxsum)
	{
		maxsum = tree->node[inode].orisum[2];
		mainori = 2;
	}
	if (tree->node[inode].orisum[3] > maxsum)
	{
		maxsum = tree->node[inode].orisum[3];
		mainori = 3;
	}
#endif
	tree->node[inode].main_ori = mainori;
	return mainori;
}

inline void SetMinLevel(CCEdgeQueue *cceq, uint8 mainori, double ori_weight)
{
	pixel minlev;
	double minlev_w;
	double minlev_ori[CONNECTIVITY / 2];
	uint8 minori;

	minlev_ori[0] = cceq->minlev_ori[0];
	minlev_ori[1] = cceq->minlev_ori[1];
#if CONNECTIVITY == 8
	minlev_ori[2] = cceq->minlev_ori[2];
	minlev_ori[3] = cceq->minlev_ori[3];
#endif
	minlev_ori[mainori] /= ori_weight;

	minlev = cceq->minlev_ori[0];
	minlev_w = minlev_ori[0];
	minori = 0;
	if (minlev_w > minlev_ori[1])
	{
		minlev = cceq->minlev_ori[1];
		minlev_w = minlev_ori[1];
		minori = 1;
	}
#if CONNECTIVITY == 8
	if (minlev_w > minlev_ori[2])
	{
		minlev = cceq->minlev_ori[2];
		minlev_w = minlev_ori[2];
		minori = 2;
	}
	if (minlev_w > minlev_ori[3])
	{
		minlev = cceq->minlev_ori[3];
		minlev_w = minlev_ori[3];
		minori = 3;
	}
#endif
	cceq->minlev = minlev;
	cceq->minori = minori;
}

void RelocCCEdgeQueue0(AlphaTree *tree, EdgeQueue *queue, CCEdgeQueue *cceq)
{
	uint8 mainori;
	uint32 lev;

	mainori = SetMainOri(tree, cceq);
	SetMinLevel(cceq, mainori, queue->ori_weight);
	if (mainori == cceq->minori)
		lev = queue->wlevelmap[cceq->minlev];
	else
		lev = queue->levelmap[cceq->minlev];
	if (cceq != queue->bucket[lev])
	{
		cceq->next = queue->bucket[lev];
		queue->bucket[lev] = cceq;
	}
	queue->minlev = MIN(queue->minlev, lev);
}
/*
uint8 testCClist(AlphaTree *tree, EdgeQueue *queue)
{
	uint32 i, j, k;
	CCEdgeQueue *p;
	for (i = 0; i < tree->imgsize * 2; i++)
		testarr[i] = 0;
	for (i = 0; i < queue->num_levels; i++)
	{
		k = 0;
		for (p = queue->bucket[i]; p; p = p->next)
		{
			j = p->iNode;
			testarr[j]++;
			if (testarr[j] == 2)
				return 1;
			k++;
		}
	}
	return 0;
}*/

void RelocCCEdgeQueue1(AlphaTree *tree, EdgeQueue *queue, CCEdgeQueue *cceq)
{
	uint8 mainori;
	uint32 lev;

	mainori = tree->node[cceq->iNode].main_ori;
	SetMinLevel(cceq, mainori, queue->ori_weight);
	if (mainori == cceq->minori)
		lev = queue->wlevelmap[cceq->minlev];
	else
		lev = queue->levelmap[cceq->minlev];
	if (queue->minlev != lev)
	{
		queue->bucket[queue->minlev] = cceq->next;
		cceq->next = queue->bucket[lev];
		queue->bucket[lev] = cceq;
		queue->minlev = MIN(queue->minlev, lev);
	}
}

void SortEdgeQueue(AlphaTree *tree, EdgeQueue *queue)
{
	CCEdgeQueue *cceq, *pNext;
	
	cceq = queue->bucket[0];
	queue->bucket[0] = NULL;
	queue->minlev = queue->num_levels;
	while (cceq)
	{
		pNext = cceq->next;
		RelocCCEdgeQueue0(tree, queue, cceq);
		cceq = pNext;
	}
}
inline bool IsEmpty(EdgeQueue *queue)
{
	return queue->minlev == queue->num_levels;
}


Edge* CCEdgeQueuePop(EdgeQueue *queue, CCEdgeQueue *cceq)
{
	Edge *ret;
	uint8 ori = cceq->minori;
	uint32 minlev, i;

	uint8 test1, test2;//tmp
	
	if (cceq->bucket[ori])
	{
		minlev = cceq->minlev_ori[ori];
		ret = cceq->bucket[ori]->levelhead[minlev];
		cceq->bucket[ori]->levelhead[minlev] = ret->next;
		if (!ret->next)
		{
			for (i = minlev + 1; i < queue->num_grey_levels && cceq->bucket[ori]->levelhead[i] == NULL; i++);
			cceq->minlev_ori[ori] = (pixel)i;
			if (i == queue->num_grey_levels)
				cceq->minlev_ori[ori] = (pixel)(queue->num_grey_levels - 1);
		}
	}
	else
	{
		ret = cceq->queue[ori];
		cceq->queue[ori] = ret->next;
		if (ret->next)
			cceq->minlev_ori[ori] = ret->next->level;
		else
			cceq->minlev_ori[ori] = (pixel)(queue->num_grey_levels - 1);

	}
	cceq->curSize[ori]--;
	numpop++;
	return ret;
}

void Phase1(AlphaTree *tree, EdgeQueue *queue, uint32 *root, const pixel *img, const pixel **ori_imgs, uint32 width, uint32 height)
{
	uint32 imgsize = width * height;
	uint32 p, x, y;
	
	for (p = 0; p < 2 * imgsize; p++)
		root[p] = p;
	CCEdgeQueueInit(queue);
	AlphaNodeInit(tree, img, ori_imgs, queue);

	tree->node[imgsize - 1].ccqueue->minori = 0;
	tree->node[imgsize - 1].ccqueue->minlev = (pixel)(queue->num_grey_levels - 1);;
	tree->node[imgsize - 1].ccqueue->next = NULL;
	tree->node[imgsize - 1].ccqueue->iNode = imgsize - 1;
	queue->bucket[0] = tree->node[imgsize - 1].ccqueue;
	queue->cnt = 1;
	p = imgsize - 2;
	for (x = 0; x < width - 1; x++) //Last row
	{
		PushEdge_NewPixel(tree, queue, p, p + 1, img, root, 0);
		p--;
	}

	for (y = 0; y < height - 1; y++)
	{
		PushEdge_NewPixel(tree, queue, p, p + width, img, root, 1);
#if CONNECTIVITY == 8
		PushEdge_OldPixel(tree, queue, p, p + width - 1, img, root, 3);
#endif
		p--;
		for (x = 0; x < width - 1; x++)
		{
			PushEdge_NewPixel(tree, queue, p, p + width, img, root, 1);
			PushEdge_OldPixel(tree, queue, p, p + 1, img, root, 0);
#if CONNECTIVITY == 8
			PushEdge_OldPixel(tree, queue, p, p + width - 1, img, root, 3);
			PushEdge_OldPixel(tree, queue, p, p + width + 1, img, root, 2);
#endif
			p--;

		}
	}

	SortEdgeQueue(tree, queue);
}

bool IsRoot(AlphaTree *tree, uint32 iNode)
{
	return tree->node[iNode].parent == iNode;
}

Edge* EdgeQueuePop(AlphaTree *tree, EdgeQueue *queue)
{
	CCEdgeQueue *p;

	while (1) //Edgequeue bucket should never be empty
	{
		p = queue->bucket[queue->minlev];
		if (p)
		{
			while (p && !IsRoot(tree, p->iNode))
				p = p->next;
			queue->bucket[queue->minlev] = p;
			if (p)
				break;			
		}
		queue->minlev++;
	}

	return CCEdgeQueuePop(queue, queue->bucket[queue->minlev]);
}

uint8 testcceqbucket(CCEdgeQueue *cceq)
{
	uint8 test0 = 0, test1 = 0, test2 = 0, test3 = 0;
	if (cceq->minlev_ori[0] < 255)
	{
		if (cceq->bucket[0])
			test0 = (cceq->bucket[0]->levelhead[cceq->minlev_ori[0]] == NULL) ||
			(cceq->bucket[0]->levelhead[cceq->minlev_ori[0]] == (Edge*)(0xcdcdcdcdcdcdcdcd));
		else
			test0 = cceq->queue[0]->level != cceq->minlev_ori[0];
	}
	if (cceq->minlev_ori[1] < 255)
	{
		if (cceq->bucket[1])
			test1 = (cceq->bucket[1]->levelhead[cceq->minlev_ori[1]] == NULL) ||
			(cceq->bucket[1]->levelhead[cceq->minlev_ori[1]] == (Edge*)(0xcdcdcdcdcdcdcdcd));
		else
			test1 = cceq->queue[1]->level != cceq->minlev_ori[1];
	}
#if CONNECTIVITY == 8
	if (cceq->minlev_ori[2] < 255)
	{
		if (cceq->bucket[2])
			test1 = (cceq->bucket[2]->levelhead[cceq->minlev_ori[2]] == NULL) ||
			(cceq->bucket[2]->levelhead[cceq->minlev_ori[2]] == (Edge*)(0xcdcdcdcdcdcdcdcd));
		else
			test1 = cceq->queue[2]->level != cceq->minlev_ori[2];
	}
	if (cceq->minlev_ori[3] < 255)
	{
		if (cceq->bucket[3])
			test1 = (cceq->bucket[3]->levelhead[cceq->minlev_ori[3]] == NULL) ||
			(cceq->bucket[3]->levelhead[cceq->minlev_ori[3]] == (Edge*)(0xcdcdcdcdcdcdcdcd));
		else
			test1 = cceq->queue[3]->level != cceq->minlev_ori[3];
	}
#endif
	//	if (cceq->bucket[0] == NULL && cceq->minlev_ori[0] < 255)
	//		return 1;
	//	if (cceq->bucket[1] == NULL && cceq->minlev_ori[1] < 255)
	//		return 1;

	return  test0 || test1 || test2 || test3;
}

void Phase2(AlphaTree *tree, EdgeQueue *queue, uint32 *root, uint32 width, uint32 height)
{
	Edge *currentEdge;
	uint32 v1, v2, r, imgsize = width * height;

	uint8 test1, test2;
	//uint32 v11, v22;
	uint32 cnt = 0;
	v1 = 0;
	v2 = 0;
	bool AreWeDoneYet = 0;
	double alpha12;
	while (!AreWeDoneYet) {
		
		if (testcceqbucket(tree->node[v2].ccqueue))
			test1 = test1;
		currentEdge = EdgeQueuePop(tree, queue); // calc minlev (using weight and stuff) later
	
		v1 = FindRoot(root, currentEdge->p);
		v2 = FindRoot(root, currentEdge->q);

		if (v1 != v2) {
			alpha12 = queue->level2alpha[queue->minlev];

			test1 = testcceqbucket(tree->node[v2].ccqueue);
		//	if (currentEdge->p == 37022 && currentEdge->q == 37323)
		//		test1 = test1;

			if (v1 > v2)
				Swap(&v1, &v2);
			if (tree->node[v2].alpha < alpha12)
			{
				r = NewAlphaNode(tree, alpha12);
				Union3(tree, queue, root, v1, v2, r);
				v2 = r;
				
			}
			else { //IF tree->node[v1].alpha > alpha12, chage alpha12 to tree->node[v1].alpha
				Union2(tree, queue, root, v1, v2);
			}
			
			test2 = testcceqbucket(tree->node[v2].ccqueue);
			if (test1 == 0 && test2 == 1)
				test1 = test1;
			if (test1 || test2)
				test1 = test1;
			AreWeDoneYet = (tree->node[v2].area == imgsize);

		}

		RelocCCEdgeQueue1(tree, queue, tree->node[v2].ccqueue);
	}
}

bool IsLevelRoot(AlphaTree *tree, uint32 i) {
	int parent = tree->node[i].parent;

	if (parent == i)
		return true;
	return (tree->node[i].alpha != tree->node[parent].alpha);
}

AlphaTree *MakeAlphaTree(const pixel *img, const pixel **ori_imgs, uint8 connectivity, uint32 width, uint32 height, uint32 channel, double ori_weight)
{
	uint32 imgsize = (uint32)width * (uint32)height;
	EdgeQueue queue;
	uint32 *root = (uint32*)Malloc(imgsize * 2 * sizeof(uint32));
	AlphaTree *tree;
//	uint32 *dhist;
//	uint8 *dimg;
//	int p;
//	double nredges;
	uint32 treesize;

//	dhist = (uint32*)Malloc(256 * sizeof(uint32));
//	dimg = (uint8*)Malloc(imgsize * 2 * sizeof(uint8));
	//memset(dhist, 0, 256 * sizeof(uint32));

	//compute_dhist(dhist, dimg, img, width, height);

	//Tree Size Estimation (TSE)
	/*
	nrmsd = 0;
	nredges = (double)(width * (height - 1) + (width - 1) * height);
	for (p = 0; p < 256; p++)
		nrmsd += ((double)dhist[p]) * ((double)dhist[p]);
	nrmsd = sqrt((nrmsd - (double)nredges) / ((double)nredges * ((double)nredges - 1.0)));
	if (mem_scheme == TSE)
		treesize = min(2 * imgsize, (uint32)(imgsize * A * (exp(SIGMA * nrmsd) + B + M)));
	else
		treesize = (uint32)(imgsize * size_init[mem_scheme]);
*/
	//don't use TSE..yet
	treesize = (uint32)(2 * imgsize);

	EdgeQueueInit(&queue, width, height, ori_weight);
	tree = CreateAlphaTree(imgsize, treesize);
	assert(tree != NULL);
	assert(tree->node != NULL);
	//fprintf(stderr,"Phase1 started\n");
	//Free(queue.levelmap);
	//Realloc(queue.levelmap, queue.num_grey_levels);//tmp
	Phase1(tree, &queue, root, img, ori_imgs, width, height);
	//fprintf(stderr,"Phase2 started\n");
	Phase2(tree, &queue, root, width, height);
	//fprintf(stderr,"Phase2 done\n");
	EdgeQueueDelete(&queue);
	Free(root);

	//Free(dhist);
	return tree;
}

pixel* SalienceTreeSalienceFilter(AlphaTree *tree, pixel *out, double lambda, uint32 size)
{
	uint32 i, imgsize = size, alpha_no;
	//out = (pixel*)malloc(size * sizeof(pixel));
	if (lambda <= tree->node[tree->curSize - 1].alpha) {
		tree->node[tree->curSize - 1].outval =
			tree->node[tree->curSize - 1].sumPix / tree->node[tree->curSize - 1].area;
		for (i = tree->curSize - 2; i != 0xffffffff; i--) {
			if (tree->node[i].alpha <= lambda) {
				alpha_no = i;
				break;
			}
		}
		for (i = tree->curSize - 2; i != 0xffffffff; i--) {
			if (tree->node[i].parent <= alpha_no) {
				tree->node[i].outval = tree->node[tree->node[i].parent].outval;
				tree->node[i].filtered = true;
			}
			else {
				tree->node[i].outval = tree->node[i].sumPix / tree->node[i].area;
			}
		}
	}
	else {
		for (i = tree->curSize - 1; i != 0xffffffff; i--) {
			tree->node[i].outval = 0;
		}
	}
	for (i = 0; i < imgsize; i++)
		out[i] = (pixel)(tree->node[i].outval + .5);
	return out;
}// Filter based on values

void SalienceTreeAreaFilter(AlphaTree *tree, pixel *out, int lambda) {
	uint32 i, imgsize = tree->maxSize / 2;
	if (lambda <= imgsize) {
		tree->node[tree->curSize - 1].outval =
			tree->node[tree->curSize - 1].sumPix / tree->node[tree->curSize - 1].area;
		for (i = tree->curSize - 2; i != 0xffffffff; i--) {

			if (IsLevelRoot(tree, i) && (tree->node[i].area >= lambda)) {
				tree->node[i].outval = tree->node[i].sumPix / tree->node[i].area;
			}
			else {
				tree->node[i].outval = tree->node[tree->node[i].parent].outval;
			}
		}
	}
	else {
		for (i = tree->curSize - 1; i != 0xffffffff; i--) {
			tree->node[i].outval = 0;
		}
	}
	for (i = 0; i < imgsize; i++)
		out[i] = (pixel)(tree->node[i].outval+0.5);
}

#define NodeSalience(tree, p) (tree->node[Par(tree,p)].alpha)
/*
void SalienceTreeSalienceFilter(SalienceTree *tree, pixel *out, double lambda) {
	int i, j, imgsize = tree->maxSize / 2;
	if (lambda <= tree->node[tree->curSize - 1].alpha) {
		for (j = 0; j < 3; j++) {
			tree->node[tree->curSize - 1].outval[j] =
				tree->node[tree->curSize - 1].sumPix[j] / tree->node[tree->curSize - 1].area;
		}
		for (i = tree->curSize - 2; i >= 0; i--) {

			if (IsLevelRoot(tree, i) && (NodeSalience(tree, i) >= lambda)) {
				for (j = 0; j < 3; j++)
					tree->node[i].outval[j] = tree->node[i].sumPix[j] / tree->node[i].area;
			}
			else {
				for (j = 0; j < 3; j++)
					tree->node[i].outval[j] = tree->node[tree->node[i].parent].outval[j];
			}
		}
	}
	else {
		for (i = tree->curSize - 1; i >= 0; i--) {
			for (j = 0; j < 3; j++)
				tree->node[i].outval[j] = 0;
		}
	}
	for (i = 0; i < imgsize; i++)
		for (j = 0; j < 3; j++)
			out[i][j] = tree->node[i].outval[j];

}
*/

/*

short ImagePPMAsciiRead(char *fname)
{
   FILE *infile;
   ulong i,j;
   int c;

   infile = fopen(fname, "r");
   if (infile==NULL) {
	  fprintf (stderr, "Error: Can't read the ASCII file: %s !", fname);
	  return(0);
   }
   fscanf(infile, "P3\n");
   while ((c=fgetc(infile)) == '#')
	  while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%d %d\n255\n", &width, &height);
   size = width*height;

   gval = malloc(size*sizeof(Pixel));
   if (gval==NULL) {
	  fprintf (stderr, "Out of memory!");
	  fclose(infile);
	  return(0);
   }
   for (i=0; i<size; i++)
   {
	 for (j=0;j<3;j++){
	   fscanf(infile, "%d", &c);
	   gval[i][j] = c;
	 }
   }
   fclose(infile);
   return(1);
}


short ImagePPMBinRead(char *fname)
{
   FILE *infile;
   int c, i;

   infile = fopen(fname, "rb");
   if (infile==NULL) {
	  fprintf (stderr, "Error: Can't read the binary file: %s !", fname);
	  return(0);
   }
   fscanf(infile, "P6\n");
   while ((c=fgetc(infile)) == '#')
	  while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%d %d\n255\n", &width, &height);
   size = width*height;

   gval = malloc(size*sizeof(Pixel));
   if (gval==NULL) {
	 fprintf (stderr, "Out of memory!");
	 fclose(infile);
	 return(0);
   }
   fread(gval, sizeof(Pixel), size, infile);

   fclose(infile);
   return(1);
}


short ImagePPMRead(char *fname)
{
   FILE *infile;
   char id[4];

   infile = fopen(fname, "r");
   if (infile==NULL) {
	  fprintf (stderr, "Error: Can't read the image: %s !", fname);
	  return(0);
   }
   fscanf(infile, "%3s", id);
   fclose(infile);
   if (strcmp(id, "P3")==0) return(ImagePPMAsciiRead(fname));
   else if (strcmp(id, "P6")==0) return(ImagePPMBinRead(fname));
   else {
	 fprintf (stderr, "Unknown type of the image!");
	 return(0);
   }
}

int ImagePPMBinWrite(char *fname)
{
   FILE *outfile;

   outfile = fopen(fname, "wb");
   if (outfile==NULL) {
	  fprintf (stderr, "Error: Can't write the image: %s !", fname);
	  return(-1);
   }
   fprintf(outfile, "P6\n%d %d\n255\n", width, height);

   fwrite(out,sizeof(Pixel) , (size_t)(size), outfile);

   fclose(outfile);
   return(0);
}
*/

int main(int argc, char *argv[]) {

	AlphaTree *tree;
	uint8 *main_ori;
	uint32 width, height, channel;
	uint32 cnt = 0;
	ofstream f;
	ifstream fcheck;
	char in;
	uint32 i, contidx;
	std::string path;

	contidx = 0;
	//	f.open("C:/Users/jwryu/RUG/2018/AlphaTree/AlphaTree_grey_Exp.dat", std::ofstream::app);
	fcheck.open(OUTPUT_FNAME);
	if (fcheck.good())
	{
		cout << "Output file \"" << OUTPUT_FNAME << "\" already exists. Overwrite? (y/n/a)";
		cin >> in;
		if (in == 'a')
		{
			f.open(OUTPUT_FNAME, std::ofstream::app);
			cout << "Start from : ";
			cin >> contidx;
		}
		else if (in == 'y')
			f.open(OUTPUT_FNAME);
		else
			exit(-1);
	}
	else
		f.open(OUTPUT_FNAME);

	cnt = 0;
	for (mem_scheme = 0; mem_scheme < 2; mem_scheme++) // memory scheme loop (TSE, max)
	{
		for (i = 0; i < 2; i++) // grey, colour loop
		{
			if (i == 0)
				path = INPUTIMAGE_DIR;
			else
				path = INPUTIMAGE_DIR_COLOUR;

			for (auto & p : std::experimental::filesystem::directory_iterator(path))
			{
				if (++cnt < contidx)
				{
					cout << cnt << ": " << p << endl;
					continue;
				}
				cv::String str1(p.path().string().c_str());
				cv::Mat cvimg;
				cv::Mat outimg;
				cv::Mat outimg1;
				if (i == 0)
					cvimg = imread(str1, cv::IMREAD_GRAYSCALE);
				else
				{
					cvimg = imread(str1, cv::IMREAD_COLOR);
					cv::cvtColor(cvimg, cvimg, CV_BGR2GRAY);
				}

				/*
				cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
				cv::imshow("Display window", cvimg);                   // Show our image inside it.
				cv::waitKey(0);
				getc(stdin);
				*/

				height = cvimg.rows;
				width = cvimg.cols;
				channel = cvimg.channels();
				
				cout << cnt << ": " << str1 << ' ' << height << 'x' << width << endl;

				if (channel != 1)
				{
					cout << "input should be a greyscale image" << endl;
					getc(stdin);
					exit(-1);
				}


				cv::Mat dest[CONNECTIVITY / 2];
				double orientation[4] = { M_PI / 2, 0, M_PI / 4, 3 * M_PI / 4 };
				cv::Mat src_f;
				cvimg.convertTo(src_f, CV_32F);

				pixel *ori_imgs[CONNECTIVITY / 2];
				for (int ori_idx = 0; ori_idx < CONNECTIVITY / 2; ori_idx++)
				{
					int kernel_size = 31;
					double sig = 1, th = orientation[ori_idx], lm = 1.0, gm = 0.02, ps = 0;
					cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm, ps);
					cv::filter2D(src_f, dest[ori_idx], CV_32F, kernel);
					dest[ori_idx].convertTo(dest[ori_idx], CV_8U, 1.0 / 255.0); //because I don't know how to access float type Mat...
					ori_imgs[ori_idx] = (pixel*)dest[ori_idx].data;
					//cerr << dest(Rect(30, 30, 10, 10)) << endl; // peek into the data

					//Gabor Filtering visualization
					/*
					cv::Mat viz;
					cv::imshow("input", cvimg);
					cv::imshow("k", kernel);
					//dest[ori_idx].convertTo(dest[ori_idx], CV_8U, 1.0 / 255.0);
					cv::imshow("d", dest[ori_idx]);
					cv::waitKey();
					*/

				}

				double runtime, minruntime;
				for (int testrep = 0; testrep < REPEAT; testrep++)
				{
					memuse = max_memuse = 0;

					/*
					main_ori = new uint8[width * height];

					for (uint32 img_itr = 0; img_itr < width * height; img_itr++)
					{
						pixel mag, max_ori_mag = dest[0].data[img_itr];
						uint32 ori_mag_sum = max_ori_mag;
						uint8 max_ori = 0;

						for (uint8 ori_idx = 1; ori_idx < CONNECTIVITY / 2; ori_idx++)
						{
							mag = dest[ori_idx].data[img_itr];
							ori_mag_sum += mag;
							if (mag > max_ori_mag)
							{
								max_ori_mag = mag;
								max_ori = ori_idx;
							}
						}
						ori_mag_sum = ori_mag_sum / (CONNECTIVITY / 2);
						if (ori_mag_sum > 10 && (double)max_ori_mag > 1.1 * ori_mag_sum)
							main_ori[img_itr] = max_ori + 1;
						else
							main_ori[img_itr] = 0;
					}					
					cv::imshow("a", cvimg);
					for (uint32 ii = 0; ii < width * height; ii++)
						cvimg.data[ii] = 127*main_ori[ii];
					cv::imshow("b", dest[0]);
					cv::imshow("c", dest[1]);
					cv::imshow("d", cvimg);
					cv::waitKey();
					*/

					/*
					pixel testimg[15] = { 255, 127, 111,95,15,255,135,79,95,63,223,135,47,47,95 };
					pixel ori[30] = { 0,8,31,0,50,30,7,32,28,40,30,0,25,43,30,
					128,65,20,25,80,117,85,40,22,32,90,85,40,25,50 };
					pixel *ori_imgs[2];
					ori_imgs[0] = ori;
					ori_imgs[1] = &ori[15];
					tree = MakeSalienceTree(testimg, ori_imgs, 4, 5, 3, 1, 2, 3.0);
					*/
					auto wcts = std::chrono::system_clock::now();

					tree = MakeAlphaTree((const pixel*)cvimg.data, (const pixel**)ori_imgs, CONNECTIVITY, width, height, channel, 1.0);
					//		start = clock();

					
					cvimg.copyTo(outimg);
					cvimg.copyTo(outimg1);
					for (uint32 i = 0; i < height * width; i++)
						outimg.data[i] = 0;
					SalienceTreeSalienceFilter(tree, (pixel*)outimg.data, 10, width * height);
					tree = MakeAlphaTree((const pixel*)cvimg.data, (const pixel**)ori_imgs, CONNECTIVITY, width, height, channel, 2.0);

					for (uint32 i = 0; i < height * width; i++)
						outimg1.data[i] = 0;
					SalienceTreeSalienceFilter(tree, (pixel*)outimg1.data, 5, width * height);

					cv::imshow("out_1.0", outimg);
					cv::imshow("out_2.0", outimg1);
					cv::waitKey();
					

					std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
					runtime = wctduration.count();
					minruntime = testrep == 0 ? runtime : min(runtime, minruntime);

					if (testrep < (REPEAT - 1))
						DeleteTree(tree);
				}
				f << p.path().string().c_str() << '\t' << height << '\t' << width << '\t' << max_memuse << '\t' << nrmsd << '\t' << tree->maxSize << '\t' << tree->curSize << '\t' << minruntime << endl;

				cout << "Time Elapsed: " << minruntime << endl;
				cvimg.release();
				str1.clear();
				DeleteTree(tree);
			}
		}
	}


	f.close();
	return 0;
} /* main */
