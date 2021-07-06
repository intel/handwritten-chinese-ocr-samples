/////////////////////////////////////////////////////////////////////////////////////////
//
// recovery a document image from *.dgr file
// Source from: http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
// Modified by bliu3650
// 
// Build example:
// g++ -std=c++11 -o dgr2png dgr2png.c 
//     -I/usr/local/include/opencv4 -L/usr/local/lib/
//     -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>


#define MAX_ILLUSTR_LEN 128

// the head information of the *.dgr file
struct DGR_HEADER {
  int iHdSize; // size of header: 54+strlen(illustr)+1 (there is a '\0' at the end of illustr)
  char szFormatCode[8]; // "DGR"
  char szIllustr[MAX_ILLUSTR_LEN]; // text of arbitrary length. "#......\0"
  char szCodeType[20]; // "ASCII", "GB", "SJIS" etc
  short sCodeLen; // 1, 2, 4, etc
  short sBitApp; // "1 or 8 bit per pixel" etc
};

// the annotation information of a word
struct WORD_INFO {
  unsigned char *pWordLabel; // the pointer to the word label (GB code)
  short sTop; // the top coordinate of a word image
  short sLeft; // the left coordinate of a word image
  short sHei; // the height of a word image
  short sWid; // the width of a word image
};

// the annotation information of a text line
struct LINE_INFO {
  int iWordNum; // the word number in a text line
  int iTop; // the top coordinate of a line image
  int iLeft; // the left coordinate of a line image
  int iBottom; // the bottom coordinate of a line image
  int iRight; // the right coordinate of a line image
  WORD_INFO *pWordInfo; // the pointer to the annotation information of the words in a text line
};

// the annotation information of document image
struct DOC_IMG {
  int iImgHei; // the height of the document image
  int iImgWid; // the width of the document image
  int iLineNum; // the text line number in the document image
  LINE_INFO *pLineInfo; // the pointer to the annotation information of the text lines
  unsigned char *pDocImg; // the pointer to image data buffer
};

/////////////////////////////////////////////////////////////////////////////////////////
// //
// read annotation information from *.dgr file //
// recovery the * dgr file to document image data //
// //
/////////////////////////////////////////////////////////////////////////////////////////
bool ReaddgrFile2Img(FILE *fp, // fp is the file pointer to *.dgr file
                     std::string &dgr_file_path,
                     std::string &extracted_data_folder,
                     std::ifstream &hwdb1x_img_gt_file,
                     std::string &alpha_symbols_file_path,
                     bool synthesize_flag,
                     std::string &img_code_end_str) {

    std::string dgr_file_name = dgr_file_path.substr(dgr_file_path.find_last_of("/") + 1);
    std::string dgr_base_name = dgr_file_name.substr(0, dgr_file_name.find_last_of("."));

    DGR_HEADER dgrHead;
    DOC_IMG docImg;

    // read the head information of the *.dgr file
    fread(&dgrHead.iHdSize, 4, 1, fp);
    fread(dgrHead.szFormatCode, 8, 1, fp);
    fread(dgrHead.szIllustr, (dgrHead.iHdSize - 36), 1, fp);
    fread(dgrHead.szCodeType, 20, 1, fp);
    fread(&dgrHead.sCodeLen, 2, 1, fp);
    fread(&dgrHead.sBitApp, 2, 1, fp);

    // read the height and width of the document image
    fread(&docImg.iImgHei, 4, 1, fp);
    fread(&docImg.iImgWid, 4, 1, fp);

    // allocate memory for the document image data
    docImg.pDocImg = new unsigned char [docImg.iImgHei * docImg.iImgWid];
    memset(docImg.pDocImg, 0xff, docImg.iImgHei * docImg.iImgWid);
    fread(&docImg.iLineNum, 4, 1, fp);
    docImg.pLineInfo = new LINE_INFO [docImg.iLineNum];

    int i, j, m, n;
    unsigned char *pTmpData;
    int iTmpDataSize;
    short iTmpDataTop;
    short iTmpDataLeft;
    short iTmpDataHei;
    short iTmpDataWid;

    printf("Total lines: %d\n", docImg.iLineNum);

    // recovery the document image line by line
    for (i = 0; i < docImg.iLineNum; i++) {
        printf("Line: %d\n", i);
        // read the word number in the i-th text line
        fread(&docImg.pLineInfo[i].iWordNum, 4, 1, fp);

        // read the annotation information of every word in the i-th text line
        //printf("Total words: %d\n", docImg.pLineInfo[i].iWordNum);
        docImg.pLineInfo[i].pWordInfo = new WORD_INFO [docImg.pLineInfo[i].iWordNum];
        docImg.pLineInfo[i].iTop = docImg.iImgHei; // initialization
        docImg.pLineInfo[i].iLeft = 0;
        docImg.pLineInfo[i].iBottom = 0;
        docImg.pLineInfo[i].iRight = 0;

        // create file to store the extracted lables (codes)
        std::string label_filename = dgr_base_name + "-L" +
            std::to_string(i+1) + img_code_end_str + ".txt";
        std::ofstream label_file;
        label_file.open(extracted_data_folder + "/" + label_filename);

        for (j = 0; j < docImg.pLineInfo[i].iWordNum; j++) {
            //printf("Word: %d\n", j);
            docImg.pLineInfo[i].pWordInfo[j].pWordLabel = new unsigned char [dgrHead.sCodeLen];
            fread(docImg.pLineInfo[i].pWordInfo[j].pWordLabel, dgrHead.sCodeLen, 1, fp);
            fread(&docImg.pLineInfo[i].pWordInfo[j].sTop, 2, 1, fp);
            fread(&docImg.pLineInfo[i].pWordInfo[j].sLeft, 2, 1, fp);
            fread(&docImg.pLineInfo[i].pWordInfo[j].sHei, 2, 1, fp);
            fread(&docImg.pLineInfo[i].pWordInfo[j].sWid, 2, 1, fp);

            // update the max top and min bottom coordinates of current line image
            if (docImg.pLineInfo[i].pWordInfo[j].sTop < docImg.pLineInfo[i].iTop) {
                docImg.pLineInfo[i].iTop = docImg.pLineInfo[i].pWordInfo[j].sTop;
            }
            if (docImg.pLineInfo[i].pWordInfo[j].sTop + docImg.pLineInfo[i].pWordInfo[j].sHei >
                docImg.pLineInfo[i].iBottom) {
                docImg.pLineInfo[i].iBottom =
                    docImg.pLineInfo[i].pWordInfo[j].sTop +
                    docImg.pLineInfo[i].pWordInfo[j].sHei;
            }

            //printf("Lable of Word %d is: 0x%02x%02x\n", j,
            //       docImg.pLineInfo[i].pWordInfo[j].pWordLabel[0],
            //       docImg.pLineInfo[i].pWordInfo[j].pWordLabel[1]);
            std::string WORD_LABEL_HEX;
            std::stringstream ssh, ssl;
            ssh << std::uppercase << std::setw(2) << std::setfill('0') << std::hex
                << (int) docImg.pLineInfo[i].pWordInfo[j].pWordLabel[0];
            std::string high_hex = ssh.str();
            ssl << std::uppercase << std::setw(2) << std::setfill('0') << std::hex
                << (int) docImg.pLineInfo[i].pWordInfo[j].pWordLabel[1];
            std::string low_hex = ssl.str();
            WORD_LABEL_HEX = high_hex + low_hex;

            // if set synthesize_flag to true but the current word is too small,
            // then do not replace this word.
            bool replace_flag = false;
            if (synthesize_flag == true) {
                std::ifstream alpha_symbols_file(alpha_symbols_file_path);
                if (alpha_symbols_file.is_open()) {
                    std::string line;
                    while (std::getline(alpha_symbols_file, line)) {
                        if (WORD_LABEL_HEX == line) {
                            replace_flag = false;
                            break;
                        }
                        replace_flag = true;
                    }
                    alpha_symbols_file.close();
                }
            }

            iTmpDataTop = docImg.pLineInfo[i].pWordInfo[j].sTop;
            iTmpDataLeft = docImg.pLineInfo[i].pWordInfo[j].sLeft;
            iTmpDataHei = docImg.pLineInfo[i].pWordInfo[j].sHei;
            iTmpDataWid = docImg.pLineInfo[i].pWordInfo[j].sWid;
            pTmpData = new unsigned char [iTmpDataHei * iTmpDataWid];
            memset(pTmpData, 0xff, iTmpDataHei * iTmpDataWid);

            if (replace_flag == false) {
                label_file << WORD_LABEL_HEX + "\n";
                fread(pTmpData, iTmpDataHei * iTmpDataWid, 1, fp);
            } 
            else {
                // replace_flag == true
                // also need to update fp
                fread(pTmpData, iTmpDataHei * iTmpDataWid, 1, fp);
                // get the image path and groudtruth for replacing the img data and label code
                std::string target_img;
                std::string img_path;
                std::string img_gt; // code
                if (!std::getline(hwdb1x_img_gt_file, target_img)) {
                    return true; // END_OF_SYNTH
                }
                std::stringstream ss(target_img);
                std::getline(ss, img_path, ',');
                std::getline(ss, img_gt, ',');

                // load and resize img to iTmpDataEdgeSize * iTmpDataEdgeSize
                cv::Mat img = cv::imread(img_path, 0);
                cv::Mat resized;
                // h==w is better for performance
			    short iTmpDataEdgeSize = iTmpDataHei < iTmpDataWid ? iTmpDataHei : iTmpDataWid;
                cv::resize(img, resized, cv::Size(iTmpDataEdgeSize, iTmpDataEdgeSize), 0, 0);
                memset(pTmpData, 0xff, iTmpDataHei * iTmpDataWid); // clear the original image
                // copy selected image to replace the original word region start from top left
				for (m = 0; m < iTmpDataEdgeSize; m++) {
                    for (n = 0; n < iTmpDataEdgeSize; n++) {
                        pTmpData[m * iTmpDataWid + n]
							= ((unsigned char*)(resized.data))[m * iTmpDataEdgeSize + n];
                    }
                }

                // replace the lable code and store it to label file
                label_file << img_gt + "\n";
            }

			// write the the word data image to the document image data
            for (m = 0; m < iTmpDataHei; m++) {
                for (n = 0; n < iTmpDataWid; n++) {
                    if (pTmpData[m * iTmpDataWid + n] != 255) {
                        docImg.pDocImg[(m + iTmpDataTop) * docImg.iImgWid + n + iTmpDataLeft]
                            = pTmpData[m * iTmpDataWid + n];
                    }
                }
            }

            delete [] pTmpData;
            delete [] docImg.pLineInfo[i].pWordInfo[j].pWordLabel;
        }

        label_file.close();

        // calculate the min left and max right coordinates of current line image
        docImg.pLineInfo[i].iLeft = docImg.pLineInfo[i].pWordInfo[0].sLeft >= 0 ?
            docImg.pLineInfo[i].pWordInfo[0].sLeft : 0;
        docImg.pLineInfo[i].iRight = 
            docImg.pLineInfo[i].pWordInfo[docImg.pLineInfo[i].iWordNum-1].sLeft +
            docImg.pLineInfo[i].pWordInfo[docImg.pLineInfo[i].iWordNum-1].sWid;
    }

    // crop the image to lines using only the words belong to them
    for (i = 0; i < docImg.iLineNum; i++) {
        //printf("crop Line: %d\n", i);
        int iHei = docImg.pLineInfo[i].iBottom - docImg.pLineInfo[i].iTop;
        int iWid = docImg.pLineInfo[i].iRight - docImg.pLineInfo[i].iLeft;
        unsigned char* pLingImg = new unsigned char [iHei * iWid];
        memset(pLingImg, 0xff, iHei * iWid);

        for (j = 0; j < docImg.pLineInfo[i].iWordNum; j++) {
            //printf("crop Word: %d\n", j);
            iTmpDataTop = docImg.pLineInfo[i].pWordInfo[j].sTop;
            iTmpDataLeft = docImg.pLineInfo[i].pWordInfo[j].sLeft;
            iTmpDataHei = docImg.pLineInfo[i].pWordInfo[j].sHei;
            iTmpDataWid = docImg.pLineInfo[i].pWordInfo[j].sWid;
            int topInLineImg = iTmpDataTop - docImg.pLineInfo[i].iTop;
            int leftInLineImg = iTmpDataLeft - docImg.pLineInfo[i].iLeft;
            for (m = 0; m < iTmpDataHei; m++) {
                for (n = 0; n < iTmpDataWid; n++) {
                    if ((m + topInLineImg) * iWid + n + leftInLineImg < iHei * iWid) {
                        pLingImg[(m + topInLineImg) * iWid + n + leftInLineImg] =
                            docImg.pDocImg[(m + iTmpDataTop) * docImg.iImgWid + n + iTmpDataLeft];
                    }
                }
            }
        }

        cv::Mat img = cv::Mat(iHei, iWid, CV_8UC1, (void *)pLingImg);
        int targetHei = 128; // fix
        int targetWid = 128; // initialization
        cv::Mat targetImg;
        if (iHei > targetHei) {
            float whRatio = (float)iWid / (float)iHei;
            targetWid = (int) (targetHei * whRatio);
            cv::resize(img, targetImg, cv::Size(targetWid, targetHei), 0, 0);
        } else {
			// put at the center of target region
            targetWid = iWid;
            targetImg = cv::Mat::ones(targetHei, targetWid, CV_8UC1) * 255;
            int h_start = (targetHei - iHei) / 2;
            cv::Mat targetImgRoi(targetImg, cv::Rect(0, h_start, targetWid, iHei));
            img.copyTo(targetImgRoi);
        }

        std::string line_img_name = dgr_base_name + "-L" +
            std::to_string(i+1) + img_code_end_str + ".png";
        cv::imwrite(extracted_data_folder + "/" + line_img_name, targetImg);

        delete [] pLingImg;
        delete [] docImg.pLineInfo[i].pWordInfo;
    }

    delete [] docImg.pLineInfo;
    delete [] docImg.pDocImg;

    return false; // not the end of synthesize
}


bool IsPathExist(const std::string &s) {
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}


int main(int argc, char *argv[]) {
    bool SYNTHESIZE_MODE = false;
    if (argc != 3 && argc != 5) {
        printf("USAGE: %s <dgr list file> <extract data folder>"
               " [hwdb1x gt file] [alpha symbol file] \n", argv[0]);
        return 1;
    }

    if (argc == 5) {
        SYNTHESIZE_MODE = true;
    }

    std::string extracted_data_folder(argv[2]);
    if (IsPathExist(extracted_data_folder) == false) {
        printf("Folder not exist.\n");
        return 1;
    }

    if (SYNTHESIZE_MODE == false) {
        std::string file_path;
        std::ifstream null_img_gt_file("");
        std::string null_alpha_symbols_file_path("");
        std::string img_code_end_str("");
        std::ifstream dgr_list_file(argv[1]);
        if (dgr_list_file.is_open() == false) {
            printf("Fail to open file.\n");
            return 1;
        }
        while (std::getline(dgr_list_file, file_path)) {
            printf("dgr file: %s \n", file_path.c_str());
            FILE * fp;
            const char *dgr_file_path = file_path.c_str();
            fp = fopen(dgr_file_path, "r");
            if (NULL == fp) {
                printf("Fail to open file.\n");
                return 1;
            }
            ReaddgrFile2Img(fp, file_path,
                extracted_data_folder,
                null_img_gt_file,
                null_alpha_symbols_file_path,
                false,
                img_code_end_str
            );
            fclose(fp);
        }
        dgr_list_file.close();
    }
    // NOTE: do not use the same name as original image for synthesized images!
    else { // SYNTHESIZE_MODE == true
        std::ifstream hwdb1x_img_gt_file(argv[3]);
        std::string alpha_symbols_file_path(argv[4]);
        if (hwdb1x_img_gt_file.is_open() == false) {
            printf("Fail to open file.\n");
            return 1;
        }
        if (IsPathExist(alpha_symbols_file_path) == false) {
            printf("File not exist.\n");
            return 1;
        }
        bool END_OF_SYNTH = false;
        int synth_iter_num = 1;
        while (!END_OF_SYNTH) {
            std::string file_path;
            std::string img_code_end_str("-S"+std::to_string(synth_iter_num));
            std::ifstream dgr_list_file(argv[1]);
            if (dgr_list_file.is_open() == false) {
                printf("Fail to open file.\n");
                return 1;
            }
            while (std::getline(dgr_list_file, file_path)) {
                printf("dgr file: %s \n", file_path.c_str());
                FILE * fp;
                const char *dgr_file_path = file_path.c_str();
                fp = fopen(dgr_file_path, "r");
                if (NULL == fp) {
                    printf("Fail to open file.\n");
                    return 1;
                }
                END_OF_SYNTH = ReaddgrFile2Img(
                    fp, file_path,
                    extracted_data_folder,
                    hwdb1x_img_gt_file,
                    alpha_symbols_file_path,
                    true,
                    img_code_end_str
                );
                fclose(fp);
                if (END_OF_SYNTH) {
                    break;
                }
            }
            dgr_list_file.close();
            synth_iter_num += 1;
        }
        hwdb1x_img_gt_file.close();
    }

    return 0;
}
