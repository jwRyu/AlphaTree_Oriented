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

#define OUTPUT_FNAME "C:/Users/jwryu/RUG/2018/AlphaTree/SalienceTree_grey_10rep.dat"

#define INPUTIMAGE_DIR	"C:/Users/jwryu/Google Drive/RUG/2018/AlphaTree/imgdata/Grey"
#define INPUTIMAGE_DIR_COLOUR	"C:/Users/jwryu/Google Drive/RUG/2018/AlphaTree/imgdata/Colour" //colour images are used after rgb2grey conversion
#define REPEAT 10

#define max(a,b) (a)>(b)?(a):(b)
#define min(a,b) (a)>(b)?(b):(a)
#define false 0
#define true  1

#define CONNECTIVITY  4
#define NUM_GREYLEVELS	256

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
/*
typedef struct Edge {
	int p, q;
	//  double alpha;
} Edge;

typedef struct {
	int maxsize;
	Edge *queue;
	uint8 *dimg;
	uint32 *bottom, *cur;
	uint32 minalpha, maxalpha;
} EdgeQueue;

EdgeQueue *EdgeQueueCreate(long maxsize, uint8 *dimg, uint32* dhist) {
	EdgeQueue *newQueue = (EdgeQueue *)Malloc(sizeof(EdgeQueue));
	uint32 sum, i;
	newQueue->queue = (Edge *)Malloc(maxsize * sizeof(Edge));
	newQueue->dimg = dimg;
	newQueue->maxsize = maxsize;
	newQueue->bottom = (uint32*)Malloc(257 * sizeof(uint32));
	newQueue->cur = (uint32*)Malloc(257 * sizeof(uint32));
	newQueue->minalpha = 256;
	newQueue->maxalpha = 255;
	sum = 0;
	for (i = 0; i < 256; i++)
	{
		newQueue->bottom[i] = newQueue->cur[i] = sum;
		sum += dhist[i];
	}
	newQueue->cur[256] = 1;
	newQueue->bottom[256] = 0;

	return newQueue;
}


//#define EdgeQueueFront(queue)       (queue->queue + 1)
#define IsEmpty(queue)        (queue->minalpha > queue->maxalpha)

void EdgeQueueDelete(EdgeQueue *oldqueue) {
	Free(oldqueue->queue);
	Free(oldqueue->bottom);
	Free(oldqueue->cur);
	Free(oldqueue->dimg);
	Free(oldqueue);
}

Edge* EdgeQueueFront(EdgeQueue *queue)
{
	return queue->queue + queue->cur[queue->minalpha] - 1;
}

void EdgeQueuePop(EdgeQueue *queue) {
	queue->cur[queue->minalpha]--;

	while (queue->cur[queue->minalpha] == queue->bottom[queue->minalpha])
		queue->minalpha++;
}

void EdgeQueuePush(EdgeQueue *queue, int p, int q, uint8 alpha) {
	uint32 idx = queue->cur[alpha]++;

	queue->queue[idx].p = p;
	queue->queue[idx].q = q;
	queue->minalpha = min(queue->minalpha, alpha);
}*/

typedef struct Edge
{
	uint32 p, q;
	pixel level;
	struct Edge *pNext;
} Edge;

typedef struct CCEdgeBucket
{
	Edge *levelhead[NUM_GREYLEVELS];
} CCEdgeBucket;

typedef struct CCEdgeQueue
{
	Edge *queue[CONNECTIVITY / 2];
	CCEdgeBucket *bucket[CONNECTIVITY / 2];
	uint32 curSize[CONNECTIVITY / 2];
	struct CCEdgeQueue *pNext;
} CCEdgeQueue;

typedef struct EdgeQueue
{
	CCEdgeQueue **bucket;
	uint32 *levelmap, *wlevelmap;
	uint32 minlev, num_grey_levels, num_levels;
	//curSize
	CCEdgeQueue *CCrepo;
	uint32 CCrepo_cursize, CCrepo_maxsize;
	Edge *Edgerepo;
	uint32 Edgerepo_cursize, Edgerepo_maxsize;
	CCEdgeBucket *bucketrepo;
	uint32 bucketrepo_cursize, bucketrepo_maxsize;
	double ori_weight;
} EdgeQueue;

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
	queue->Edgerepo_maxsize = 2 * (img_width * (img_height - 1) + (img_width - 1) * img_height);
	queue->bucketrepo_maxsize = (uint32)((queue->Edgerepo_maxsize + NUM_GREYLEVELS - 1) / NUM_GREYLEVELS);
	queue->CCrepo = (CCEdgeQueue*)Malloc(queue->CCrepo_maxsize * sizeof(CCEdgeQueue));
	queue->Edgerepo = (Edge*)Malloc(queue->Edgerepo_maxsize * sizeof(Edge));
	queue->bucketrepo = (CCEdgeBucket*)Malloc(queue->bucketrepo_maxsize * sizeof(CCEdgeBucket));

	level = (double*)Malloc(NUM_GREYLEVELS * sizeof(double));
	wlevel = (double*)Malloc(NUM_GREYLEVELS * sizeof(double));
	for (i = 0; i < NUM_GREYLEVELS; i++)
	{
		level[i] = (double)i;
		wlevel[i] = ((double)i) / ori_weight;
	}

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
	Free(queue->levelmap);
	Free(queue->wlevelmap);
	Free(queue);
}

typedef struct AlphaNode
{
	uint32 parent;
	uint32 area;
	bool filtered; /* indicates whether or not the filtered value is OK */
	pixel outval;  /* output value after filtering */
	uint8 alpha;  /* alpha of flat zone */
	double sumPix;
	pixel minPix;
	pixel maxPix;
	double orisum[CONNECTIVITY / 2];
	uint8 main_ori; // 0: hor / 1: ver / 2: diag(135deg) / 3: diag(45) / 4: none or insignificant (lower than thr)
} AlphaNode;



typedef struct AlphaTree {
	int imgsize;
	int maxSize, curSize;
	AlphaNode *node;
} AlphaTree;


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


int NewAlphaNode(AlphaTree *tree, uint32 *root, uint8 alpha) {
	AlphaNode *node = tree->node + tree->curSize;
	int result;
	if (tree->curSize == tree->maxSize)
	{
		printf("Reallocating...\n");
		tree->maxSize = min(tree->imgsize * 2, tree->maxSize + (int)(tree->imgsize * size_add[mem_scheme]));

		tree->node = (AlphaNode*)Realloc(tree->node, tree->maxSize * sizeof(AlphaNode));
		node = tree->node + tree->curSize;
	}

	result = tree->curSize;
	tree->curSize++;
	node->area = 0;
	node->alpha = alpha;
	node->parent = BOTTOM;
	root[result] = BOTTOM;
	return result;
}

inline void MakeSet(AlphaTree *tree, uint32 *root, pixel *gval, uint32 p) {
	int i;
	tree->node[p].parent = p;
	root[p] = p;
	tree->node[p].alpha = 0;
	tree->node[p].area = 1;
	for (i = 0; i < 3; i++) {
		tree->node[p].sumPix = gval[p];
	}
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

int FindRoot1(AlphaTree *tree, int *root, int p) {
	int r = p, i, j;

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

inline uint8 LinfNormX(pixel *img,
	int width,
	int height,
	int x, int y) {

	int p = width * y + x - 1, q = width * y + x;

	return (uint8)abs((int)img[p] - (int)img[q]);
}

inline uint8 LinfNormY(pixel *img,
	int width,
	int height,
	int x, int y) {

	int p = width * (y - 1) + x, q = width * y + x;

	return (uint8)abs((int)img[p] - (int)img[q]);
}

bool IsLevelRoot(AlphaTree *tree, int i) {
	int parent = tree->node[i].parent;

	if (parent == BOTTOM)
		return true;
	return (tree->node[i].alpha != tree->node[parent].alpha);
}

int LevelRoot(AlphaTree *tree, int p) {
	int r = p, i, j;

	while (!IsLevelRoot(tree, r)) {
		r = tree->node[r].parent;
	}
	i = p;

	while (i != r) {
		j = tree->node[i].parent;
		tree->node[i].parent = r;
		i = j;
	}
	return r;
}

# define Par(tree,p) LevelRoot(tree,tree->node[p].parent)

void Union(AlphaTree *tree, int *root, int p, int q) { /* p is always current pixel */
	q = FindRoot1(tree, root, q);

	if (q != p) {
		tree->node[q].parent = p;
		root[q] = p;
		tree->node[p].area += tree->node[q].area;
		tree->node[p].sumPix += tree->node[q].sumPix;
		tree->node[p].minPix = MIN(tree->node[p].minPix, tree->node[q].minPix);
		tree->node[p].maxPix = MAX(tree->node[p].maxPix, tree->node[q].maxPix);
	}
}


void Union2(AlphaTree *tree, int *root, int p, int q) {
	tree->node[q].parent = p;
	root[q] = p;
	tree->node[p].area += tree->node[q].area;
	tree->node[p].sumPix += tree->node[q].sumPix;
	tree->node[p].minPix = MIN(tree->node[p].minPix, tree->node[q].minPix);
	tree->node[p].maxPix = MAX(tree->node[p].maxPix, tree->node[q].maxPix);
}

void Union3(AlphaTree *tree, int *root, int p, int q, int r) {
	int i;
	tree->node[p].parent = r;
	tree->node[q].parent = r;
	root[p] = r;
	root[q] = r;
	tree->node[r].area = tree->node[p].area + tree->node[q].area;
	tree->node[r].sumPix = tree->node[p].sumPix + tree->node[q].sumPix;
	tree->node[r].minPix = MIN(tree->node[p].minPix, tree->node[q].minPix);
	tree->node[r].maxPix = MAX(tree->node[p].maxPix, tree->node[q].maxPix);
}

void compute_dhist(uint32 *dhist, uint8 *dimg, pixel *img, int width, int height) {
	/* pre: tree has been created with imgsize= width*height
	queue initialized accordingly;
	*/
	uint32 imgsize = width * height;
	uint32 p, x, y;
	uint32 dimgidx;
	uint8 edgeSalience;

	dimgidx = 3;
	for (x = 1; x < width; x++) {
		edgeSalience = LinfNormX(img, width, height, x, 0);
		dimg[dimgidx] = edgeSalience;
		dhist[edgeSalience]++;
		dimgidx += 2;
	}
	dimgidx--;

	for (y = 1; y < height; y++) {
		p = y * width;
		edgeSalience = LinfNormY(img, width, height, 0, y);
		dimg[dimgidx] = edgeSalience;
		dhist[edgeSalience]++;
		dimgidx += 2;

		p++;
		for (x = 1; x < width; x++, p++) {
			edgeSalience = LinfNormY(img, width, height, x, y);
			dimg[dimgidx++] = edgeSalience;
			dhist[edgeSalience]++;


			edgeSalience = LinfNormX(img, width, height, x, y);
			dimg[dimgidx++] = edgeSalience;
			dhist[edgeSalience]++;
		}
	}
}

void Phase1(AlphaTree *tree, EdgeQueue *queue, uint32 *root,
	pixel *img, uint32 width, uint32 height) {
	/* pre: tree has been created with imgsize= width*height
	queue initialized accordingly;
	*/
	uint32 imgsize = width * height;
	uint32 p, x, y;
	//uint32 dimgidx;

	uint8 edgeSalience;

	MakeSet(tree, root, img, 0);

	for (x = 1; x < width; x++) {
		MakeSet(tree, root, img, x);
		edgeSalience = dimg[dimgidx];
		dimgidx += 2;
		if (edgeSalience == 0)
			Union(tree, root, x, x - 1);
		else
			EdgeQueuePush(queue, x, x - 1, edgeSalience);
	}
	dimgidx--;

	for (y = 1; y < height; y++) {
		p = y * width;
		MakeSet(tree, root, img, p);
		edgeSalience = dimg[dimgidx];
		dimgidx += 2;

		if (edgeSalience == 0)
			Union(tree, root, p, p - width);
		else
			EdgeQueuePush(queue, p, p - width, edgeSalience);

		p++;
		for (x = 1; x < width; x++, p++) {
			MakeSet(tree, root, img, p);
			edgeSalience = dimg[dimgidx++];

			if (edgeSalience == 0)
				Union(tree, root, p, p - width);
			else
				EdgeQueuePush(queue, p, p - width, edgeSalience);

			edgeSalience = dimg[dimgidx++];

			if (edgeSalience == 0)
				Union(tree, root, p, p - 1);
			else
				EdgeQueuePush(queue, p, p - 1, edgeSalience);
		}

	}

}

void GetAncestors(AlphaTree *tree, int *root, int *p, int*q) {
	int temp;
	*p = LevelRoot(tree, *p);
	*q = LevelRoot(tree, *q);
	if (*p < *q) {
		temp = *p;
		*p = *q;
		*q = temp;
	}
	while ((*p != *q) && (root[*p] != BOTTOM) && (root[*q] != BOTTOM)) {
		*q = root[*q];
		if (*p < *q) {
			temp = *p;
			*p = *q;
			*q = temp;
		}
	}
	if (root[*p] == BOTTOM) {
		*q = FindRoot(root, *q);
	}
	else if (root[*q] == BOTTOM) {
		*p = FindRoot(root, *p);
	}
}

void Phase2(AlphaTree *tree, EdgeQueue *queue, int *root,
	pixel *img,
	int width, int height) {
	Edge *currentEdge;
	int v1, v2, temp, r;
	uint8 oldalpha = 0, alpha12;
	while (!IsEmpty(queue)) {
		currentEdge = EdgeQueueFront(queue);
		v1 = currentEdge->p;
		v2 = currentEdge->q;
		GetAncestors(tree, root, &v1, &v2);
		alpha12 = queue->minalpha;

		EdgeQueuePop(queue);
		if (v1 != v2) {
			if (v1 < v2) {
				temp = v1;
				v1 = v2;
				v2 = temp;
			}
			if (tree->node[v1].alpha < alpha12) {
				r = NewAlphaNode(tree, root, alpha12);
				Union3(tree, root, v1, v2, r);
			}
			else {
				Union2(tree, root, v1, v2);
			}
		}
		oldalpha = alpha12;
	}
}

AlphaTree *MakeSalienceTree(pixel *img, pixel **ori_imgs,
	uint32 width, uint32 height, uint32 channel, double ori_weight, double lambdamin) {
	uint32 imgsize = (uint32)width * (uint32)height;
	EdgeQueue queue;
	uint32 *root = (uint32*)Malloc(imgsize * 2 * sizeof(uint32));
	AlphaTree *tree;
	uint32 *dhist;
	uint8 *dimg;
	int p;
	double nredges;
	int treesize;


	dhist = (uint32*)Malloc(256 * sizeof(uint32));
	dimg = (uint8*)Malloc(imgsize * 2 * sizeof(uint8));
	memset(dhist, 0, 256 * sizeof(uint32));

	//compute_dhist(dhist, dimg, img, width, height);

	//Tree Size Estimation (TSE)
	nrmsd = 0;
	nredges = (double)(width * (height - 1) + (width - 1) * height);
	for (p = 0; p < 256; p++)
		nrmsd += ((double)dhist[p]) * ((double)dhist[p]);
	nrmsd = sqrt((nrmsd - (double)nredges) / ((double)nredges * ((double)nredges - 1.0)));
	if (mem_scheme == TSE)
		treesize = min(2 * imgsize, (uint32)(imgsize * A * (exp(SIGMA * nrmsd) + B + M)));
	else
		treesize = (uint32)(imgsize * size_init[mem_scheme]);

	//don't use TSE..yet
	treesize = (uint32)(imgsize);

	EdgeQueueInit(&queue, width, height, ori_weight);
	tree = CreateAlphaTree(imgsize, treesize);
	assert(tree != NULL);
	assert(tree->node != NULL);
	//fprintf(stderr,"Phase1 started\n");
	Phase1(tree, &queue, root, img, width, height, lambdamin);
	//fprintf(stderr,"Phase2 started\n");
	Phase2(tree, &queue, root, img, width, height);
	//fprintf(stderr,"Phase2 done\n");
	EdgeQueueDelete(&queue);
	Free(root);

	Free(dhist);
	return tree;
}

void SalienceTreeAreaFilter(AlphaTree *tree, pixel *out, int lambda) {
	int i, j, imgsize = tree->maxSize / 2;
	if (lambda <= imgsize) {
		tree->node[tree->curSize - 1].outval =
			tree->node[tree->curSize - 1].sumPix / tree->node[tree->curSize - 1].area;
		for (i = tree->curSize - 2; i >= 0; i--) {

			if (IsLevelRoot(tree, i) && (tree->node[i].area >= lambda)) {
				tree->node[i].outval = tree->node[i].sumPix / tree->node[i].area;
			}
			else {
				tree->node[i].outval = tree->node[tree->node[i].parent].outval;
			}
		}
	}
	else {
		for (i = tree->curSize - 1; i >= 0; i--) {
			tree->node[i].outval = 0;
		}
	}
	for (i = 0; i < imgsize; i++)
		out[i] = tree->node[i].outval;
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

				double runtime, minruntime;
				for (int testrep = 0; testrep < REPEAT; testrep++)
				{
					memuse = max_memuse = 0;
					auto wcts = std::chrono::system_clock::now();

					cv::Mat dest[CONNECTIVITY / 2];
					double orientation[4] = { 0, M_PI / 2, M_PI / 4, 3 * M_PI / 4 };
					cv::Mat src_f;
					cvimg.convertTo(src_f, CV_32F);

					for (int ori_idx = 0; ori_idx < CONNECTIVITY / 2; ori_idx++)
					{
						int kernel_size = 31;
						double sig = 1, th = orientation[ori_idx], lm = 1.0, gm = 0.02, ps = 0;
						cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm, ps);
						cv::filter2D(src_f, dest[ori_idx], CV_32F, kernel);
						dest[ori_idx].convertTo(dest[ori_idx], CV_8U, 1.0 / 255.0); //because I don't know how to access float type Mat...
						//cerr << dest(Rect(30, 30, 10, 10)) << endl; // peek into the data

						//Gabor Filtering visualization
						/*cv::Mat viz;
						cv::imshow("input", cvimg);
						cv::imshow("k", kernel);
						dest[ori_idx].convertTo(dest[ori_idx], CV_8U, 1.0 / 255.0);
						cv::imshow("d", dest[ori_idx]);
						cv::waitKey();
						*/

					}
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
					/*
					cv::imshow("a", cvimg);
					for (uint32 ii = 0; ii < width * height; ii++)
						cvimg.data[ii] = 127*main_ori[ii];
					cv::imshow("b", dest[0]);
					cv::imshow("c", dest[1]);
					cv::imshow("d", cvimg);
					cv::waitKey();
					*/

					pixel testimg[15] = { 255, 127, 111,95,15,255,135,79,95,63,223,135,47,47,95 };
					pixel ori[30] = { 0,8,31,0,50,30,7,32,28,40,30,0,25,43,30,
					128,65,20,25,80,117,85,40,22,32,90,85,40,25,50 };
					pixel *ori_imgs[CONNECTIVITY / 2];
					ori_imgs[0] = ori;
					ori_imgs[1] = &ori[15];
					tree = MakeSalienceTree(testimg, ori_imgs, 5, 3, 1, 3.0);

					//					tree = MakeSalienceTree(cvimg.data, width, height, channel, 3.0);
										//		start = clock();



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
