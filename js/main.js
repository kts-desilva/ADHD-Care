/**
 * Created by User on 9/8/2019.
 */
var fmri_formdata;
var eye_movement_formdata;
$('#fmri_data_file').change(function(){
    //on change event
    fmri_formdata = new FormData();
    if($(this).prop('files').length > 0){
        file =$(this).prop('files')[0];
        fmri_formdata.append("file", file);
    }
    console.log(fmri_formdata);
});

$('#eye_movement_data_file').change(function(){
    //on change event
    eye_movement_formdata = new FormData();
    if($(this).prop('files').length > 0){
        file =$(this).prop('files')[0];
        eye_movement_formdata.append("file", file);
    }
    console.log(eye_movement_formdata);
});

function fmri_predict_value() {
    $('#fmri_loading_image').show();
    $('#fmri_predict_button ').enable(false);
    console.log(fmri_formdata);
    $.ajax({
        url:'/fmri_uploader',
        type:'POST',
        data:fmri_formdata,
        processData: false,
        contentType: false,
        success:function (res) {
            console.log(res);
            // document.getElementById('fmri_adhd_val').innerHTML+=res.adhd;
            // document.getElementById('fmri_nadhd_val').innerHTML+=res.nadhd;
            // if(parseInt(res.adhd)>parseInt(res.nadhd)){
            //     document.getElementById('fmri_diagnosis').innerHTML+='ADHD';
            //     document.getElementById('fmri_diagnosis').class+=' text-danger';
            // }else{
            //     document.getElementById('fmri_diagnosis').innerHTML+='Non-ADHD';
            //     document.getElementById('fmri_diagnosis').class+=' text-success';
            // }
        },
        complete: function(){
            $('#fmri_loading_image').hide();
            $('#fmri_predict_button ').enable(true);
        },

    });
}

function eye_movement_predict_value() {
    $('#eye_movement_loading_image').show();
    console.log(fmri_formdata);
    $.ajax({
        url:'/em_uploader',
        type:'POST',
        data:fmri_formdata,
        processData: false,
        contentType: false,
        success:function (res) {
            console.log(res);
            document.getElementById('eye_movement_adhd_val').innerHTML+=res.adhd;
            document.getElementById('eye_movement_nadhd_val').innerHTML+=res.nadhd;
        },
        complete: function(){
            $('#eye_movement_loading_image').hide();
        },

    });
}