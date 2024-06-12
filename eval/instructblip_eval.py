import os, pathlib, json, pdb
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
import textwrap
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

MODEL_NAME = "Salesforce/instructblip-vicuna-7b"
# MODEL_NAME = "Salesforce/instructblip-vicuna-13b"
print(MODEL_NAME)

processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

device = "cuda"
model.to(device)

def render_text(text):
    image = Image.new("RGB", (1024,1024), (255,255,255))
    font = ImageFont.truetype("./ctx_cfq/OpenSans-Regular.ttf", 25)
    draw = ImageDraw.Draw(image)

    text_wrapped = textwrap.wrap(text, width=80)
    text_wrapped = '\n'.join(text_wrapped)
    draw.text((10,10), text_wrapped, (0,0,0), font=font)
    return image

def model_with_image_text_input(image_path, text, input_image_type='path'):
    if input_image_type=='path':
        image = Image.open(image_path)
    elif input_image_type=='direct':
        image = image_path

    inputs = processor(image, text=text, return_tensors="pt").to(device, torch.float16)
    
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=64
    )
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return response

def infer_true_or_false(response):
    normalized_response = response.lower().replace('\n', ' ').replace(',', '').replace('.', '').split(' ')
    for word in normalized_response:
        if word == 'true':
            return 'True'
        elif word == 'false':
            return 'False'
    return 'NA'




if __name__ == '__main__':
    model_name = 'instructblip_pred'
    model_keys = {
        'InstructBLIP': 'instructblip_pred'
    }
    
    prompt_type = 'commonsense_TF_reversed-simple_TF_reversed-simple_TF_reversed-simple_TF_reversed'
    pt_a, pt_b, pt_c, pt_d = prompt_type.split('-')




    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", dest="data_path", type=str, action="store", default='../data/data.json')
    parser.add_argument("-op", "--output_path", dest="output_path", type=str, action="store", default='../data/pred/data_instructblip_7b.json')
    parser.add_argument("-id", "--image_dir", dest="image_dir", type=str, action="store", default='../data/images/counterfactual')
    parser.add_argument("-fid", "--factual_image_dir", dest="factual_image_dir", type=str, action="store", default='../data/images/factual')
    parser.add_argument("-vt", "--vote_thres", dest="vote_thres", type=int, action="store", default=2)
    args = parser.parse_args()


    def prompt(statement, prompt_type):
        if prompt_type == 'detailed_TF':
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Forget real-world common sense and just follow the information provided in the image. Only respond in True or False."
        if prompt_type == 'simple_TF':
            return f"Statement: {statement}\nBased on the image, is the given statement true or false? Only respond in True or False."
        if prompt_type == 'simple_TF_reversed':
            return f"Only respond in True or False.\nStatement: {statement}\nBased on the image, is the given statement true or false?"
        if prompt_type == 'commonsense_TF_reversed':
            return f"Only respond in True or False.\nStatement: {statement}\nBased on common sense, is the given statement true or false?"
        raise ValueError('invlaid prompt type!')
    
    def ctx_prompt(context, statement, prompt_type):
        if prompt_type == 'detailed_TF':
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Forget real-world common sense and just follow the information provided in the context. Only respond in True or False."
        if prompt_type == 'simple_TF':
            return f"Context: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false? Only respond in True or False."
        if prompt_type == 'simple_TF_reversed':
            return f"Only respond in True or False.\nContext: {context}\nStatement: {statement}\nBased on the context, is the given statement true or false?"
        raise ValueError('invlaid prompt type!')

    def obj_det_statement_template(obj):
        return f"There is {obj} in the given image."


    with open(args.data_path, 'r') as f:
        context_list = json.load(f)
    
    blank_image_path = './ctx_cfq/blank.png'
    

    for i, instance in enumerate(tqdm(context_list)):
        good_images = []

        for image_id, vote in instance['aggregated_human_label_good_images'].items():
            if vote >= args.vote_thres:
                good_images.append(image_id)

        if len(good_images) == 0:
            continue

        concept = instance['concept']
        context = instance['context']
        factual_context = instance['factual_context']
        true_statement = instance['true_statement']
        false_statement = instance['false_statement']

        input_a1 = prompt(false_statement, pt_a) # True
        input_a2 = prompt(true_statement, pt_a) # False

        input_b1 = prompt(obj_det_statement_template(instance['existent_noun']), pt_b) # True
        input_b2 = prompt(obj_det_statement_template(instance['non-existent_noun']), pt_b) # False

        input_c1 = ctx_prompt(context, true_statement, pt_c) # True
        input_c2 = ctx_prompt(context, false_statement, pt_c) # False

        input_d1 = prompt(true_statement, pt_d) # True
        input_d2 = prompt(false_statement, pt_d) # False



        input_a1_rendering_image = render_text(input_a1)
        input_a1_rendering_image_path = './ctx_cfq/temp_a1.png'
        input_a1_rendering_image.save(input_a1_rendering_image_path)

        input_a2_rendering_image = render_text(input_a2)
        input_a2_rendering_image_path = './ctx_cfq/temp_a2.png'
        input_a2_rendering_image.save(input_a2_rendering_image_path)


        input_c1_rendering_image = render_text(input_c1)
        input_c1_rendering_image_path = './ctx_cfq/temp_c1.png'
        input_c1_rendering_image.save(input_c1_rendering_image_path)

        input_c2_rendering_image = render_text(input_c2)
        input_c2_rendering_image_path = './ctx_cfq/temp_c2.png'
        input_c2_rendering_image.save(input_c2_rendering_image_path)

        for key, model_name in model_keys.items():

            image_dir = args.image_dir + f"/{instance['concept']}/{instance['context_id']}_{instance['context']}"
            factual_image_dir = args.factual_image_dir + f"/{instance['concept']}/{instance['context_id']}_{instance['factual_context']}"

            best_cf_image_id = instance['best_img_id']

            counterfactual_image_path = image_dir + f'/{best_cf_image_id}.jpg'
            factual_image_path = factual_image_dir + '/0.jpg'


            instance[model_name + '_a1_image'] = {'response': model_with_image_text_input(factual_image_path, input_a1)}
            instance[model_name + '_a2_image'] = {'response': model_with_image_text_input(factual_image_path, input_a2)}
            # instance[model_name + '_a1_blank'] = {'response': model_with_image_text_input(blank_image_path, input_a1)}
            # instance[model_name + '_a2_blank'] = {'response': model_with_image_text_input(blank_image_path, input_a2)}
            # instance[model_name + '_a1_render'] = {'response': model_with_image_text_input(input_a1_rendering_image_path, input_a1)}
            # instance[model_name + '_a2_render'] = {'response': model_with_image_text_input(input_a2_rendering_image_path, input_a2)}

            instance[model_name + '_a1_image']['answer'] = infer_true_or_false(instance[model_name + '_a1_image']['response'])
            instance[model_name + '_a2_image']['answer'] = infer_true_or_false(instance[model_name + '_a2_image']['response'])
            # instance[model_name + '_a1_blank']['answer'] = infer_true_or_false(instance[model_name + '_a1_blank']['response'])
            # instance[model_name + '_a2_blank']['answer'] = infer_true_or_false(instance[model_name + '_a2_blank']['response'])
            # instance[model_name + '_a2_render']['answer'] = infer_true_or_false(instance[model_name + '_a2_render']['response'])
            # instance[model_name + '_a1_render']['answer'] = infer_true_or_false(instance[model_name + '_a1_render']['response'])


            instance[model_name + '_b1'] = {'response': model_with_image_text_input(counterfactual_image_path, input_b1)}
            instance[model_name + '_b2'] = {'response': model_with_image_text_input(counterfactual_image_path, input_b2)}

            instance[model_name + '_b1']['answer'] = infer_true_or_false(instance[model_name + '_b1']['response'])
            instance[model_name + '_b2']['answer'] = infer_true_or_false(instance[model_name + '_b2']['response'])


            instance[model_name + '_c1_image'] = {'response': model_with_image_text_input(counterfactual_image_path, input_c1)}
            instance[model_name + '_c2_image'] = {'response': model_with_image_text_input(counterfactual_image_path, input_c2)}
            # instance[model_name + '_c1_blank'] = {'response': model_with_image_text_input(blank_image_path, input_c1)}
            # instance[model_name + '_c2_blank'] = {'response': model_with_image_text_input(blank_image_path, input_c2)}
            # instance[model_name + '_c1_render'] = {'response': model_with_image_text_input(input_c1_rendering_image_path, input_c1)}
            # instance[model_name + '_c2_render'] = {'response': model_with_image_text_input(input_c2_rendering_image_path, input_c2)}

            instance[model_name + '_c1_image']['answer'] = infer_true_or_false(instance[model_name + '_c1_image']['response'])
            instance[model_name + '_c2_image']['answer'] = infer_true_or_false(instance[model_name + '_c2_image']['response'])
            # instance[model_name + '_c1_blank']['answer'] = infer_true_or_false(instance[model_name + '_c1_blank']['response'])
            # instance[model_name + '_c2_blank']['answer'] = infer_true_or_false(instance[model_name + '_c2_blank']['response'])
            # instance[model_name + '_c1_render']['answer'] = infer_true_or_false(instance[model_name + '_c1_render']['response'])
            # instance[model_name + '_c2_render']['answer'] = infer_true_or_false(instance[model_name + '_c2_render']['response'])


            instance[model_name + '_d1'] = {}
            instance[model_name + '_d2'] = {}
            for good_image_id in good_images:
                image_path = image_dir + f'/{good_image_id}.jpg'

                instance[model_name + '_d1'][good_image_id] = {'response': model_with_image_text_input(image_path, input_d1)}
                instance[model_name + '_d2'][good_image_id] = {'response': model_with_image_text_input(image_path, input_d2)}

                instance[model_name + '_d1'][good_image_id]['answer'] = infer_true_or_false(instance[model_name + '_d1'][good_image_id]['response'])
                instance[model_name + '_d2'][good_image_id]['answer'] = infer_true_or_false(instance[model_name + '_d2'][good_image_id]['response'])


            with open(args.output_path, 'w') as f:
                json.dump(context_list, f, indent=4)