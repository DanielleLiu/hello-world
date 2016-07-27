<%@ page contentType="text/html;charset=UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="fn" uri="http://java.sun.com/jsp/jstl/functions"%>
<%@ taglib prefix="spring" uri="http://www.springframework.org/tags"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<c:set var="ctx" value="${pageContext.request.contextPath}" />
<c:set var="entityPath" value="/aaCompany" />

<html>
<head>
<script type="text/javascript" src="${ctx}/static/components/jquery-easyui-1.4/jquery.min.js"></script>
<script type="text/javascript" src="${ctx}/static/components/jquery/validation/jquery.validate.min.js" type="text/javascript"></script>
<script type="text/javascript" src="${ctx}/static/components/jquery/validation/messages_zh.js"></script>
<link rel="stylesheet" type="text/css" href="${ctx}/static/components/jquery-easyui-1.4/themes/default/easyui.css">
<link rel="stylesheet" type="text/css" href="${ctx}/static/components/jquery-easyui-1.4/themes/icon.css">
<script type="text/javascript" src="${ctx}/static/components/jquery-easyui-1.4/jquery.easyui.min.js"></script>
<script type="text/javascript" src="${ctx}/static/js/validate_.js"></script> 
	



</head>
<body>

	<div class="result">
		<form id="editform" name="editform"  action="${ctx}${entityPath}/save" method="post">

			<fieldset>
				<legend>企业基本信息</legend>
				<table
					style="font-size: 12px; font-weight: bold;  color: #666666; border-collapse: collapse; border: #D9D9D9 1px solid; width:100%;">
				<input type="hidden" name="id" id="id" value="${aaCompany.id}"/>
				
					<tr style="border: #D9D9D9 1px solid;">
						<td class="left" style="width: 20%;">企业名称</td>
						<td><input type="text" name="name" id="name" class="easyui-textbox" required="true" missingMessage="名称不能为空" value="${aaCompany.name}"/>
						</td>
					</tr>
                    <tr style="border: #D9D9D9 1px solid;">
                        <td class="left" style="width: 20%;">企业代码</td>
						<td><input type="text" id="code" class="easyui-validatebox easyui-textbox" validType="length[0,2]" invalidMessage="编码不能超过2位" name="code"  value="${aaCompany.code}" />
                        <!-- max length 如何限制input length和type -->
                        </td>
                    </tr>
					<tr style="border: #D9D9D9 1px solid;">
						<td class="left" style="width: 20%;">企业地址</td>
						<td><input type="text" name="address" id="address" class="easyui-textbox" required="true" missingMessage="地址不能为空" value="${aaCompany.address}" />
						</td>
					</tr>

					<tr style="border: #D9D9D9 1px solid;" >
					<td  style="border: #D9D9D9 1px solid;" >企业类型</td>
					<td   colspan="3"    style="border: #D9D9D9 1px solid;" >
						<input type="text" id="type" class="easyui-combotree" name="type" 
			             data-options="valueField:'id',textField:'text',editable:false,url:'${ctx}/entregtype/getAvailableParents?id=${empty aaCompany.type ? '0' : aaCompany.type}',method:'get',required:false,value:'${empty aaCompany.type ? '0' : aaCompany.type}'" />
					</td>	
					</tr>
					
					<tr style="border: #D9D9D9 1px solid;" >
					<td  style="border: #D9D9D9 1px solid;" >企业类型2</td>
					<td   colspan="3"    style="border: #D9D9D9 1px solid;" >
					<input id="type1" name="type1" class="easyui-combobox" 
					data-options=" valueField:'id',textField:'text',editable:false,data:[{'id':'0','text':'国有企业'},{'id':'1','text':'私营企业'}],value:'${aaCompany.type1}'" style="width:160px;"/>
					</td>
					</tr>

					<tr style="border: #D9D9D9 1px solid;">
						<td class="left" style="width: 20%;">座机号</td>
						<td><input type="text" name="phone" id="phone" class="easyui-validatebox easyui-textbox" data-options="validType:'phoneNo'" value="${aaCompany.phone}" />
						</td>
					</tr>				
					<tr style="border: #D9D9D9 1px solid;">
						<td class="left" style="width: 20%;">邮政编码</td>
						<td><input type="text" name="postcode" id="postcode" class="easyui-validatebox easyui-textbox" data-options="validType:'zip'" value="${aaCompany.postcode}"  />
						</td>
					</tr>
					<tr style="border: #D9D9D9 1px solid;">
						<td class="left" style="width: 20%;">注册人</td>
						<td><input type="text" name="owner" id="owner" class="easyui-validatebox easyui-textbox" validType="length[0,10]" invalidMessage="注册人姓名不能超过10位"  value="${aaCompany.owner}" />
						</td>
					</tr>
					<!-- tr style="border: #D9D9D9 1px solid;">
						<td class="left" style="width: 20%;">注册时间</td>
						<td><input type="text" name="time" id="time" class="easyui-textbox editable:false" value="${aaCompany.time}" />
						</td>
					</tr-->
					
				</table>
			</fieldset>
			<br>
			
			<div style="float: left;margin-left: 200px">
				<a id="submitlink" href="#" class="easyui-linkbutton"
					data-options="iconCls:'icon-save'" onclick="submitEditForm()">保存</a>
					
<!-- 				<a id="submitlink" href="#" class="easyui-linkbutton"
					data-options="iconCls:'icon-save'"><input     type="submit" value="bmit" />	</a>		 -->
				
				
				<a href="#" class="easyui-linkbutton"
					data-options="iconCls:'icon-back'" onclick="gosearch()">返回</a>
			</div>
		</form>
	</div>

<script type="text/javascript">

	/* $('#editform').form({
	    url:'${ctx}${entityPath}/save',
	    onClick:function(){
	       	$()
	    	
	    },
	    success:function(data){
	    	alert("success");
			window.location='${ctx}${entityPath}/list';
	    }
	    }); */
</script>

	

<!-- script type="text/javascript">
function submitEditForm(){
$("#editform").form('submit',{
	url:'${ctx}${entityPath}/edit'
	onSubmit: function(){
		$("#editform").form('validate');
	},
	success:function(data){
		alert("success");
		//document.getElementById("editform").submit()
		}
});
}
</script-->



<script type="text/javascript">
/*$(function(){
    $('#submitlink').bind('click', function(){
    	$("#editform").form({
    		function(){
    			return $("#editform").form('validate');
    		})
    });
});
*/
</script>
	
</body>
</html>