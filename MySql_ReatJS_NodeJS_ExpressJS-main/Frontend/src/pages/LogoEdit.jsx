import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './FileUploadStyle.css';
import { useParams } from 'react-router-dom';
import Sidebar from '../components/Sidebar';



// pantone color and HEX object
const colorsLib = {
    "PMS123": "#FDB927",
    "PMS268": "#552583",
    "Black": "#000000",
    "PMS5605C": "#203731",
    "PMS1235C": "#FFB612",
    "GOLD872C": "#ffd700",
    "PMS872C": "#ffd700",
    "PMS187C": "#EF3340",
    "PMS5535C": "#203731",
  

    "PMS202C": "#98002e",
    "PMS137C": "#f9a01b",
    "PMS123C": "#ffc72c",
    "PMSBlackC": "#000000",
    "PMS293C": "#006bb6",
    "PMS199C": "#ed174c",
    "PMS289C": "#002b5c",
    "PMS877C": "#c4ced4",
    "PMS268C": "#5a2b81",
    "PMS431C": "#63727a",
    "PMS282C": "#002d62",
    "PMSCoolGray5C": "#bec0c2",
    "PMS186C": "#c8102e",
    "PMS7687C": "#1d428a",
    "PMS356C": "#008348",
    "PMS874C": "#bb9753",
    "PMS174C": "#a73832",
    "PMS472C": "#fab383",
    "PMS2388C": "#0053bc",
    "yellow": "#ffc72c",
    "YELLOW": "#ffc72c",
    "PURPLE": "#552583",
    "white": "#FFFFFF",
    "WHITE": "#FFFFFF",
    "BLACK": "#000000"
  };

function LogoEdit() {
    const [data, setData] = useState(null); 
    const [gname, setGname] = useState('');
    const [logodesc, setDescription] = useState('');
    const [tname, setTname] = useState('');
    const [width, setWidth] = useState('');
    const [height, setHeight] = useState('');
    const [c1, setC1] = useState('');
    const [c2, setC2] = useState('');
    const [c3, setC3] = useState('');
    const [c4, setC4] = useState('');
    const [c5, setC5] = useState('');
    const [c6, setC6] = useState('');
    const [file, setFile] = useState(null);
    const [id, setId] = useState(0);
    const { id: RouterId } = useParams(); // Access the id parameter from the URL

    const handleFile = (e) => {
        setFile(e.target.files[0]);
    };

    useEffect(() => {
        fetchData();
        setId(RouterId); // Set the id state using the value from the route parameter
        console.log("Logo Edit page current id is: ", RouterId);
    }, [RouterId]); // Add RouterId as a dependency for useEffect

    const fetchData = () => {
        axios.get(`http://localhost:3030/logoCurrent/${RouterId}/`)
            .then(res => {
                if (res.data) {
                    setData(res.data);
                    setGname(res.data.gname);
                    setDescription(res.data.logodesc);
                    setTname(res.data.tname);
                    setWidth(res.data.width);
                    setHeight(res.data.height);
                  
                    setC1(res.data.c1);
                    setC2(res.data.c2);
                    setC3(res.data.c3);
                    setC4(res.data.c4);
                    setC5(res.data.c5);
                    setC6(res.data.c6);
                    setId(res.data.id);
                    setFile(res.data.image);
                }
            })
            .catch(err => console.log(err));
    };

    // wrapping the new users' input data to update to database
    const handleUpdate = () => {
        const formData = new FormData();
        formData.append('gname', gname);
        formData.append('logodesc', logodesc);
        formData.append('tname', tname);
        formData.append('width', width);
        formData.append('height', height);
        formData.append('c1', c1);
        formData.append('c2', c2);
        formData.append('c3', c3);
        formData.append('c4', c4);
        formData.append('c5', c5);
        formData.append('c6', c6);
        formData.append('imagel', file);

        axios.put(`http://localhost:3030/updatelogo/${id}/`, formData)
            .then(res => {
                if (res.data.Status === "Success") {
                    console.log("Update successful");
                    fetchData();
                   
                } else {
                    console.log("Update failed");
                }
            })
            .catch(err => console.log("Error during update frontend: ", err));
    };


    function colorRec(color) {
        if (color != null) {
            // Create the rectangle element
            const colorBox = (
                <div className='colorBoxSmall' style={{ backgroundColor: colorsLib[color], color: 'white' }}>
                    {/* {color} */}
                </div>
            );
            return colorBox;
        } else {
            return null; 
        }
    }
    

    return (
        <Sidebar>  
        {/* LogoEdit Page */}
        <div className='tshirt_container'>
          <div className='input_container'> 
            <h2 style={{ color: "green", margin: '10px 0 30px' }}> Edit Logo Info </h2>
            <label htmlFor="gname">Graphic #: </label>
            <input type='text' placeholder='Graphic Name' value={gname} onChange={(e) => setGname(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="logodesc">Description: </label>
            <input type='text' placeholder='Description' value={logodesc} onChange={(e) => setDescription(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="tname">Team Name: </label>
            <input type='text' placeholder='Team Name' value={tname} onChange={(e) => setTname(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="width">Width: </label>
            <input type='text' placeholder='Width' value={width} onChange={(e) => setWidth(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="height">Height: </label>
            <input type='text' placeholder='Height' value={height} onChange={(e) => setHeight(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="c1">C1: </label>
            <input type='text' placeholder='C1' value={c1} onChange={(e) => setC1(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="c2">C2: </label>
            <input type='text' placeholder='C2' value={c2} onChange={(e) => setC2(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="c3">C3: </label>
            <input type='text' placeholder='C3' value={c3} onChange={(e) => setC3(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="c4">C4: </label>
            <input type='text' placeholder='C4' value={c4} onChange={(e) => setC4(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="c5">C5: </label>
            <input type='text' placeholder='C5' value={c5} onChange={(e) => setC5(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="c6">C6: </label>
            <input type='text' placeholder='C6' value={c6} onChange={(e) => setC6(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="imageInput">Choose Image:</label>
            <input id="imageInput" type='file' onChange={handleFile} style={{ margin: '15px 0 40px 0', marginLeft: '200px'}}/> 
            <button onClick={handleUpdate} className='btn-update'> Update  </button>  <br/>
          </div>

          <div className='display_container'> 
              {data && (
                  <div>
                    <br/>
                    {data.imagel && <img src={`http://localhost:3030/images/${data.imagel}`} alt="Uploaded" style={{ width: "350px", height: "300px" }} />}
                      <br/> <br/>
                      <p><h4> Graphic #: {data.gname} </h4></p>    
                      <p><h4> Description: {data.logodesc} </h4></p>            
                      <p><h4> Team Name: {data.tname} </h4></p>
                      <p><h4> Width: {data.width} </h4></p>
                      <p><h4> Height: {data.height} </h4></p>
                      <p><h4> C1: {data.c1} </h4></p>
                      <p><h4> C2: {data.c2} </h4></p>
                      <p><h4> C3: {data.c3} </h4></p>
                      <p><h4> C4: {data.c4} </h4></p>
                      <p><h4> C5: {data.c5} </h4></p>
                      <p><h4> C6: {data.c6} </h4></p>

                      <tr className='colorRextr'> 
                        <td className='colorRextd'> {data.c1} <div> { colorRec(data.c1) } </div> </td>
                        <td className='colorRextd'> {data.c2} <div> { colorRec(data.c2) } </div> </td>
                        <td className='colorRextd'> {data.c3} <div> { colorRec(data.c3) } </div> </td>
                        <td className='colorRextd'> {data.c4} <div> { colorRec(data.c4) } </div> </td>
                        <td className='colorRextd'> {data.c5} <div> { colorRec(data.c5) } </div> </td>
                        <td className='colorRextd'> {data.c6} <div> { colorRec(data.c6) } </div> </td>
                    </tr>

                      {/* <button >Edit</button>
                      <button >Delete</button> */}
                  </div>  
              )}
          </div> 
        </div>
        </Sidebar>
    );
}

export default LogoEdit;
